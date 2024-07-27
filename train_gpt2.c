#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <assert.h>

#define DEBUG 1

FILE* fopenCheck(const char *filename, const char *mode)
{
    FILE *file = fopen(filename, mode);
    if (file == NULL) {
        printf("Error: could not open file %s\n", filename);
        exit(1);
    }
    return file;
}

void freadCheck(void *ptr, size_t size, size_t nmemb, FILE *stream)
{
    size_t nread = fread(ptr, size, nmemb, stream);
    printf("Read %zu elements\n", nread);
    if (nread != nmemb) {
        printf("Error: read error\n");
        exit(1);
    }
}

void fcloseCheck(FILE *stream)
{
    if (fclose(stream) != 0) {
        printf("Error: could not close file\n");
        exit(1);
    }
}

typedef struct
{
    int block_size; // maxT
    int vocab_size; // V
    int n_layer; // L
    int n_head; // NH
    int n_embd; // C (channel, depth of the data, embedded information)
    int padded_vocab_size; // Vp
    int batch_size; // B
} GPT2Config;

#define NUM_PARAMS 16
typedef struct {
    float* wte_w; // Vp * C
    float* wpe_w; // maxT * C
    float* ln1_w; // L * C
    float* ln1_b; // L * C
    float* attn_w; // L * 3C * C
    float* attn_b; // L * 3C
    float* attn_proj_w; // L * C * C
    float* attn_proj_b; // L * C
    float* ln2_w; // L * C
    float* ln2_b; // L * C
    float* mlp_w; // L * 4C * C
    float* mlp_b; // L * 4C
    float* mlp_proj_w; // L * 4C * C
    float* mlp_proj_b; // L * C
    float* lnf_w; // C
    float* lnf_b; // C
} ParameterTensors;

#define NUM_ACTIVATIONS 26
typedef struct {
    float* embeddings; // B * T * C
    float* ln1_mean; // L * B * T
    float* ln1_std; // L * B * T
    float* ln1_norm; // L * B * T * C
    float* ln1; // L * B * T * C
    float* qkv; // L * B * T * 3C
    float* atten_pre_softmax; // L * B * NH * T * T
    float* atten_post_softmax; // L * B * NH * T * T
    float* atten_matmul; // L * B * NH * T * T lmao my notes saids hs which is just T
    float* atten_proj; // L * B * T * C
    float* residual1; // L * B * T * C
    float* ln2_mean; // L * B * T
    float* ln2_std; // L * B * T
    float* ln2_norm; // L * B * T * C
    float* ln2; // L * B * T * C
    float* mlp; // L * B * T * 4C
    float* GELU; // L * B * T * 4C
    float* mlp_proj; // L * B * T * C
    float* residual2; // L * B * T * C
    float* lnf_mean; // B * T
    float* lnf_std; // B * T
    float* lnf_norm; // B * T * C
    float* lnf; // B * T * C
    float* decoding; // B * T * V
    float* softmax; // B * T * V
    float* cross_entropy; // B * T
} ActivationTensors;

typedef struct {
    GPT2Config config;

    ParameterTensors param_tensors;
    size_t totalparams;
    size_t param_sizes[NUM_PARAMS];
    float* param_memory;
    
    ActivationTensors activations;
    size_t totalactivations;
    size_t activation_sizes[NUM_ACTIVATIONS];
    float* activation_memory;
} GPT2;

void fill_in_parameter_sizes(size_t* param_sizes, GPT2Config config)
{
    size_t Vp = config.padded_vocab_size;
    size_t maxT = config.block_size;
    size_t L = config.n_layer;
    size_t C = config.n_embd;

    param_sizes[0] = Vp * C;         // wte_w
    param_sizes[1] = maxT * C;       // wpe_w
    param_sizes[2] = L * C;          // ln1_w
    param_sizes[3] = L * C;          // ln1_b
    param_sizes[4] = L * 3 * C * C;  // attn_w
    param_sizes[5] = L * 3 * C;      // attn_b
    param_sizes[6] = L * C * C;      // attn_proj_w
    param_sizes[7] = L * C;          // attn_proj_b
    param_sizes[8] = L * C;          // ln2_w
    param_sizes[9] = L * C;          // ln2_b
    param_sizes[10] = L * 4 * C * C; // mlp_w
    param_sizes[11] = L * 4 * C;     // mlp_b
    param_sizes[12] = L * 4 * C * C; // mlp_proj_w
    param_sizes[13] = L * C;         // mlp_proj_b
    param_sizes[14] = C;             // lnf_w
    param_sizes[15] = C;             // lnf_b
}

float* malloc_and_point_parameters(ParameterTensors* param_memory, size_t* param_sizes, size_t totalparams) {
    float* pparams = malloc(sizeof(float) * totalparams);
    float** ptr[NUM_PARAMS] = {
        &param_memory->wte_w,
        &param_memory->wpe_w,
        &param_memory->ln1_w,
        &param_memory->ln1_b,
        &param_memory->attn_w,
        &param_memory->attn_b,
        &param_memory->attn_proj_w,
        &param_memory->attn_proj_b,
        &param_memory->ln2_w,
        &param_memory->ln2_b,
        &param_memory->mlp_w,
        &param_memory->mlp_b,
        &param_memory->mlp_proj_w,
        &param_memory->mlp_proj_b,
        &param_memory->lnf_w,
        &param_memory->lnf_b
    };

    float* pparams_iter = pparams;
    for (int i = 0; i < NUM_PARAMS; i++) {
        printf("Allocating %zu bytes for parameter %d\n", param_sizes[i] * sizeof(float), i);
        *ptr[i] = pparams_iter;
        pparams_iter += param_sizes[i];
    }
    return pparams;
}

void fill_in_activation_sizes(size_t* activation_sizes, GPT2Config config, int B, int T)
{
    size_t L = config.n_layer;
    size_t C = config.n_embd;
    size_t NH = config.n_head;
    size_t Vp = config.padded_vocab_size;
    printf("L: %zu, C: %zu, NH: %zu, Vp: %zu\n", L, C, NH, Vp);

    activation_sizes[0] = B * T * C;         // embeddings
    activation_sizes[1] = L * B * T;         // ln1_mean
    activation_sizes[2] = L * B * T;         // ln1_std
    activation_sizes[3] = L * B * T * C;     // ln1_norm
    activation_sizes[4] = L * B * T * C;     // ln1
    activation_sizes[5] = L * B * T * 3 * C; // qkv
    activation_sizes[6] = L * B * NH * T * T; // atten_pre_softmax
    activation_sizes[7] = L * B * NH * T * T; // atten_post_softmax
    activation_sizes[8] = L * B * NH * T * T; // atten_matmul
    activation_sizes[9] = L * B * T * C;     // atten_proj
    activation_sizes[10] = L * B * T * C;    // residual1
    activation_sizes[11] = L * B * T;        // ln2_mean
    activation_sizes[12] = L * B * T;        // ln2_std
    activation_sizes[13] = L * B * T * C;    // ln2_norm
    activation_sizes[14] = L * B * T * C;    // ln2
    activation_sizes[15] = L * B * T * 4 * C; // mlp
    activation_sizes[16] = L * B * T * 4 * C; // GELU
    activation_sizes[17] = L * B * T * C;    // mlp_proj
    activation_sizes[18] = L * B * T * C;    // residual2
    activation_sizes[19] = B * T;            // lnf_mean
    activation_sizes[20] = B * T;            // lnf_std
    activation_sizes[21] = B * T * C;        // lnf_norm
    activation_sizes[22] = B * T * C;        // lnf
    activation_sizes[23] = B * T * Vp;        // decoding
    activation_sizes[24] = B * T * Vp;        // softmax
    activation_sizes[25] = B * T;            // cross_entropy
}

float* malloc_and_point_activations(ActivationTensors* activation_memory, size_t* activation_sizes, size_t totalactivations) {
    float* pactivations = malloc(sizeof(float) * totalactivations);
    float** ptr[NUM_ACTIVATIONS] = {
        &activation_memory->embeddings,
        &activation_memory->ln1_mean,
        &activation_memory->ln1_std,
        &activation_memory->ln1_norm,
        &activation_memory->ln1,
        &activation_memory->qkv,
        &activation_memory->atten_pre_softmax,
        &activation_memory->atten_post_softmax,
        &activation_memory->atten_matmul,
        &activation_memory->atten_proj,
        &activation_memory->residual1,
        &activation_memory->ln2_mean,
        &activation_memory->ln2_std,
        &activation_memory->ln2_norm,
        &activation_memory->ln2,
        &activation_memory->mlp,
        &activation_memory->GELU,
        &activation_memory->mlp_proj,
        &activation_memory->residual2,
        &activation_memory->lnf_mean,
        &activation_memory->lnf_std,
        &activation_memory->lnf_norm,
        &activation_memory->lnf,
        &activation_memory->decoding,
        &activation_memory->softmax,
        &activation_memory->cross_entropy
    };

    float* pactivations_iter = pactivations;
    for (int i = 0; i < NUM_ACTIVATIONS; i++) {
        printf("Allocating %zu bytes for activation %d\n", activation_sizes[i] * sizeof(float), i);
        *ptr[i] = pactivations_iter;
        pactivations_iter += activation_sizes[i];
    }
    return pactivations;
}

//load GPT2 checkpoint in karpathy format
void gpt2_build_from_checkpoint(GPT2 *model, const char *checkpoint_path)
{
    /*
        # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
        assert dtype in {"float32", "bfloat16"} # float16 todo maybe later
        version = {
            "float32": 3, # 3: all tensors are fp32, padded vocab
            "bfloat16": 5, # 5: all tensors are bf16, padded vocab
        }[dtype]
        header = torch.zeros(256, dtype=torch.int32)
        header[0] = 20240326 # magic
        header[1] = version # checkpoint version
        header[2] = model.config.block_size
        header[3] = model.config.vocab_size
        header[4] = model.config.n_layer
        header[5] = model.config.n_head
        header[6] = model.config.n_embd
        # 2) the parameters follow the header
        params = {name: param.cpu() for name, param in model.named_parameters()}
        # pad the vocab to a multiple of 128 here at export, for efficiency in C
        wte = params["transformer.wte.weight"] # (V, C)
        wte_padded = pad_vocab(wte) # (Vp, C)
        params["transformer.wte.weight"] = wte_padded # (Vp, C)
        print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
        header[7] = wte_padded.size(0) # padded vocab size store in header
        # now write to file
        with open(filename, "wb") as file:
            file.write(header.numpy().tobytes()) # header
            write_tensors(params, model.config.n_layer, file, dtype) # params
        print(f"wrote {filename}")
    */  
    
    // open and check the checkpoint file
    FILE *checkpoint_file = fopen(checkpoint_path, "rb");
    if (checkpoint_file == NULL)
    {
        printf("Error: could not open checkpoint file\n");
        return;
    }
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, checkpoint_file);
    if (model_header[0] != 20240326)
    {
        printf("Bad magic model file\n");
        exit(1);
    }
    if (model_header[1] != 3)
    {
        printf("Bad version in model file\n");
        printf("---> HINT: try to re-run `python train_gpt2.py`\n");
        exit(1);
    }

    // setup the model configuration
    model->config.block_size = model_header[2];
    model->config.vocab_size = model_header[3];
    model->config.n_layer = model_header[4];
    model->config.n_head = model_header[5];
    model->config.n_embd = model_header[6];
    model->config.padded_vocab_size = model_header[7];
    printf("GPT2 model configuration:\n");
    printf("block_size: %d\n", model->config.block_size);
    printf("vocab_size: %d\n", model->config.vocab_size);
    printf("n_layer: %d\n", model->config.n_layer);
    printf("n_head: %d\n", model->config.n_head);
    printf("n_embd: %d\n", model->config.n_embd);
    printf("padded_vocab_size: %d\n", model->config.padded_vocab_size);

    // allocate memory for the model parameters and read them in
    fill_in_parameter_sizes(model->param_sizes, model->config);
    
    model->totalparams = 0;
    for (int i = 0; i < NUM_PARAMS; i++)
    {
        model->totalparams += model->param_sizes[i];
    }
    printf("Total parameters: %zu\n", model->totalparams);

    model->param_memory = malloc_and_point_parameters(&model->param_tensors, model->param_sizes, model->totalparams);
    freadCheck(model->param_memory, sizeof(float), model->totalparams, checkpoint_file);
    fcloseCheck(checkpoint_file);
}

void gpt2_free(GPT2 *model)
{
    free(model->param_memory);
    free(model->activation_memory);
}

typedef struct {
    FILE* file;
    int T; 
    int B; 
} Dataloader;

void dataloader_init(Dataloader* loader, const char* tokens, int B, int T) {
    loader->file = fopen(tokens, "rb");
    if (loader->file == NULL) {
        printf("Error: could not open file %s\n", tokens);
        exit(1);
    }
    loader->T = T;
    loader->B = B;
}

void gpt2_forward();
void encoder_forward();

void gpt2_forward(GPT2 *model, int* input, int* targets, int B, int T)
{
    // convenience parameters
    int C = model->config.n_embd;
    int Vp = model->config.padded_vocab_size;
    int L = model->config.n_layer;
    int NH = model->config.n_head;
    int V = model->config.vocab_size;
    printf("------C: %d, Vp: %d, L: %d, NH: %d, V: %d\n", C, Vp, L, NH, V);

    // Checks
    if (model->param_memory == NULL)
    {
        printf("Error: model parameters not loaded Line: %d\n", __LINE__);
        exit(1);
    }
    // validate inputs, all indices must be in the range [0, V)
    for(int i = 0; i < B * T; i++) {
        assert(0 <= input[i] && input[i] < V);
        if (targets != NULL) {
            assert(0 <= targets[i] && targets[i] < V);
        }
    }

    // allocate activations
    if (model->activation_memory)
    {
        printf("Allocating activations\n");
        fill_in_activation_sizes(model->activation_sizes, model->config, B, T);
        model->totalactivations = 0;
        for (int i = 0; i < NUM_ACTIVATIONS; i++)
        {
            model->totalactivations += model->activation_sizes[i];
        }
        printf("Total activations: %zu\n", model->totalactivations);
        model->activation_memory = malloc_and_point_activations(&model->activations, model->activation_sizes, model->totalactivations);
    }


    // input is a 2D array of shape (B, T)
    encoder_forward(model, input, B, T);

    for (int l = 0; l < model->config.n_layer; l++)
    {
        // attention

        // mlp
    }
}

void encoder_forward(GPT2 *model, int* input, int B, int T)
{
    int C = model->config.n_embd;
    
    for (int i = 0; i < B; i ++)
    {
        for (int j = 0; j < T; j++)
        {
            for (int k = 0; k < C; k++)
            {
                model->activations.embeddings[i * T * C + j * C + k] = 0.0;
            }
        }
    }

}



int main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M_hf.bin");
    
    // dataloading
    const char* shakespeare = "input.txt";
    int B = 4;
    int T = 64;
    Dataloader train_loader, val_loader;
    dataloader_init(&train_loader, shakespeare, B, T);

    int* input = malloc(sizeof(int) * B * T);
    int* targets = malloc(sizeof(int) * B * T);
    // 1..B*T for now for testing
    for (int i = 0; i < B; i++)
    {
        for (int j = 0; j < T; j++)
        {
            input[i * T + j] = i * T + j + 1 + 200;
            targets[i * T + j] = i * T + j + 2 + 200;
        }
    }

    gpt2_forward(&model, input, targets, B, T);


    gpt2_free(&model);
    return 0;
}