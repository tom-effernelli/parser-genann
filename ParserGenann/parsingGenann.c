#include "genann.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

const int nb_epoch = 1000;
const float learning_rate = 0.05;
const char *separators = " ,.;:-/*^&=()[]|_{}+~<>\n";
const char *model_file = "model.ann";
const char *embedding_file = "glove.6B.50d.txt";

#define MAX_TOKENS 50
#define MAX_TOKEN_SIZE 25
#define VECTOR_SIZE 50
#define HASH_TABLE_SIZE 400000
#define NB_INPUTS 50
#define NB_HIDDEN_LAYERS 1
#define NB_NEURONS_PER_HIDDENS_LAYERS 64
#define NB_OUTPUTS 1

/* Training dataset relative */
const char *tokens_file = "tokens.txt";
const char *labels_file = "labels.txt";
/* 1000 = max number of lines inside training dataset, it just needs to be greater than the number of lines it actually contains */
double labels_table[1000][MAX_TOKENS];
double tokens_table[1000][MAX_TOKENS][MAX_TOKEN_SIZE];
/*
Labels matching :
 - 0 : not relevant token
 - 1 : B-PERSON (the person's name beginning)
 - 2 : I-PERSON (inside of the person's name)
 - 3 : B-LEGAL_ISSUE (the legal problem's beginning)
 - 4 : I-LEGAL_ISSUE (inside of the legal problem)
*/

typedef struct TokenEmbedding {
    char *word;
    double vector[VECTOR_SIZE];
    struct TokenEmbedding* next;
} TokenEmbedding;

typedef struct {
    TokenEmbedding** table;
} HashTable;

unsigned long hash(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++))
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    return hash % HASH_TABLE_SIZE;
}

HashTable* create_hash_table() {
    HashTable* ht = malloc(sizeof(HashTable));
    ht->table = calloc(HASH_TABLE_SIZE, sizeof(TokenEmbedding*));
    printf("Hash Table created\n");
    return ht;
}

void insert_hash(HashTable* ht, TokenEmbedding* embedding) {
    TokenEmbedding* token = malloc(sizeof(TokenEmbedding));
    unsigned long idx = hash(embedding->word);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        token->vector[i] = embedding->vector[i];
    }
    token->word = strdup(embedding->word);
    token->next = ht->table[idx];
    ht->table[idx] = token;
}

double* get_hash_vector(HashTable* ht, const char* word) {
    unsigned long idx = hash(word);
    TokenEmbedding* curr = ht->table[idx];
    while (curr) {
        if (strcmp(curr->word, word) == 0)
            return curr->vector;
        curr = curr->next;
    }
    return NULL; // not found in hash table
}

void free_hash_table(HashTable* ht) {
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        TokenEmbedding* curr = ht->table[i];
        while (curr) {
            TokenEmbedding* next = curr->next;
            free(curr);
            curr = next;
        }
    }
    free(ht->table);
    free(ht);

    printf("Hash table freed\n");
}

void load_embeddings(HashTable* ht) {
    FILE* file = fopen(embedding_file, "r");
    if (!file) {
        perror("Error while opening embedding file!");
        exit(EXIT_FAILURE);
    }

    char line[4096]; /* Number way too large for the max of a line, just to be sure its all readen */

    while (fgets(line, sizeof(line), file)) {
        char* token = strtok(line, " ");
        if (!token) continue;

        TokenEmbedding* embedding = malloc(sizeof(TokenEmbedding));
        if (!embedding) {
            perror("Error while creating embedding structure (malloc)!\n");
            exit(EXIT_FAILURE);
        }

        embedding->word = strdup(token); // copier le mot
        if (!embedding->word) {
            perror("Error while copying token (strdup)!\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < VECTOR_SIZE; i++) {
            token = strtok(NULL, " ");
            if (!token) {
                perror("Error, incomplete line inside embedding file");
                exit(EXIT_FAILURE);
            }
            embedding->vector[i] = strtof(token, NULL);
        }

        insert_hash(ht, embedding);
    }

    fclose(file);

    printf("Embeddings loaded\n");
}

void embedding(HashTable* ht, char input[], double embedding[MAX_TOKENS][VECTOR_SIZE]) {
    int embedding_count = 0;
    char *token = strtok(input, separators);

    while (token != NULL && embedding_count < MAX_TOKENS) {
        double* hash_vector = get_hash_vector(ht, strlwr(token));
            for (int i = 0; i<VECTOR_SIZE; i++) {
                embedding[embedding_count][i] = hash_vector[i];
            }

        embedding_count++;
        token = strtok(NULL, separators);
    }
    if (token != NULL){
        printf("Watch out, your request is exceeding the %d tokens limit!", MAX_TOKENS);
    }
}

void load_training_set(HashTable* ht) {
    FILE *labels = fopen(labels_file, "r");
    FILE *tokens = fopen(tokens_file, "r");

    if (!labels || !tokens) {
        perror("Error while opening labels or tokens file!\n");
        exit(EXIT_FAILURE);
    }

    /* 2048 = mex number of characters per line of tokens/labels files */
    char line_tokens[2048];
    char line_labels[2048];
    char* token_int_ptr = NULL;
    char* label_int_ptr = NULL;

    int num_lines = 0;

    while (fgets(line_tokens, sizeof(line_tokens), tokens) && 
           fgets(line_labels, sizeof(line_labels), labels)) {
        int token_index = 0;
        char *token = strtok_r(line_tokens, " \n", &token_int_ptr);
        char *label = strtok_r(line_labels, " \n", &label_int_ptr);

        while (token && label && token_index < MAX_TOKENS) {
            double* hash_vector = get_hash_vector(ht, strlwr(token));
            for (int i = 0; i<VECTOR_SIZE; i++) {
                tokens_table[num_lines][token_index][i] = hash_vector[i];
            }

            labels_table[num_lines][token_index] = atoi(label);

            label = strtok_r(NULL, " \n", &label_int_ptr);
            token = strtok_r(NULL, " \n", &token_int_ptr);
            token_index++;
        }

        num_lines++;
    }

    fclose(labels);
    fclose(tokens);

    printf("Training set loaded correctly!\n");
}

genann* init_model() {
    genann* ann = genann_init(NB_INPUTS, NB_HIDDEN_LAYERS, NB_NEURONS_PER_HIDDENS_LAYERS, NB_OUTPUTS);
    if (!ann){
        perror("Error when initializing model!");
        exit(EXIT_FAILURE);
    }
    return ann;
}

genann* load_model() {
    FILE *in = fopen(model_file, "r");
    if (!in) {
        perror("Error when opening model in reading mode!");
        exit(EXIT_FAILURE);
    }
    genann* ann = genann_read(in);
    fclose(in);
    return ann;
}

void train_model(genann* ann) {
    FILE *out = fopen(model_file, "w");
    if (!out) {
        perror("Error when opening model in writing mode!");
        exit(EXIT_FAILURE);
    }

    for (int epoch = 0; epoch < nb_epoch; ++epoch) {
        int l = 0;
        while (tokens_table[l][0][0]){
            int w = 0;
            while (tokens_table[l][w][0]){
                genann_train(ann, tokens_table[l][w], &labels_table[l][w], learning_rate);
                w++;
            }
            l++;
        }
        }
    genann_write(ann, out);

    fclose(out);
}

void free_model(genann* ann) {
    genann_free(ann);
}

/* First argument of the call is a bool to set training to true or false, the second is the prompt */
int main(int argc, char *argv[])
{
    HashTable* ht = create_hash_table();
    load_embeddings(ht);

    load_training_set(ht);

    genann* ann;
    FILE* model = fopen(model_file,"r" );
    fseek (model, 0, SEEK_END);
    int size = ftell(model);
    if (0 == size) {
      ann = init_model();
    } else {
      ann = load_model();
    }

    if (1) { /* Replace 1 by argv[1] */
        train_model(ann);
    } else {
        double embeddings[MAX_TOKENS][VECTOR_SIZE];
        char input[] = "Hi, my name is Olivia Brown and I'm looking for legal help for workplace harassment."; /* Replace this string by  argv[2] */
        embedding(ht, input, embeddings);
    
        int i = 0;
        double* results;
        while (embeddings[i][0]) {
            results[i] = *genann_run(ann, embeddings[i]);
            i++;
        }

        int j=0;
        while(results[j]) {
            printf("%f ", results[j]);
        }
    }

    free_model(ann);
    free_hash_table(ht);
    return 0;
}