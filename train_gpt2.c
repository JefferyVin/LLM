#include <stdio.h>
#include <stdlib.h>

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
    //printf("Read %zu elements\n", nread);
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

typedef struct {
    GPT2Config config;
    ParameterTensors param_tensors;
    float* params;
    size_t param_sizes[NUM_PARAMS];
    size_t totalparams;
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

float* malloc_and_point_parameters(ParameterTensors* params, size_t* param_sizes, size_t totalparams) {
    float* pparams = malloc(sizeof(float) * totalparams);
    float** ptr[NUM_PARAMS] = {
        &params->wte_w,
        &params->wpe_w,
        &params->ln1_w,
        &params->ln1_b,
        &params->attn_w,
        &params->attn_b,
        &params->attn_proj_w,
        &params->attn_proj_b,
        &params->ln2_w,
        &params->ln2_b,
        &params->mlp_w,
        &params->mlp_b,
        &params->mlp_proj_w,
        &params->mlp_proj_b,
        &params->lnf_w,
        &params->lnf_b
    };

    float* pparams_iter = pparams;
    for (int i = 0; i < NUM_PARAMS; i++) {
        printf("Allocating %zu bytes for parameter %d\n", param_sizes[i] * sizeof(float), i);
        *ptr[i] = pparams_iter;
        pparams_iter += param_sizes[i];
    }
    return pparams;
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

    model->params = malloc_and_point_parameters(&model->param_tensors, model->param_sizes, model->totalparams);
    freadCheck(model->params, sizeof(float), model->totalparams, checkpoint_file);
    fcloseCheck(checkpoint_file);
}

int main() {

    // build the GPT-2 model from a checkpoint
    GPT2 model;
    gpt2_build_from_checkpoint(&model, "gpt2_124M.bin");
    
}