import json
import os

import torch

from models.multimodal_encoder.t5_encoder import T5Embedder

MODEL_PATH = "google/t5-v1_1-xxl"

# Modify this to your task name and instruction
TASK_NAME = "put_puzzle"

INSTRUCTION = {
    "put_puzzle": {
        "instruction": "grap the pieces on the desk into the puzzle box through corresponding shapes with right arm",
        "simplified_instruction": "Put the pieces on the desk into the puzzle box through corresponding shapes with right arm",
        "expanded_instruction": "Utilize the right arm of the robot to pick up the small pieces of wood and position them in accordance with their distinct shapes on the wooden box. If the small block gets stuck, gently touch the edge of the shape to make it fall along the shape."
    }
}

# Note: if your GPU VRAM is less than 24GB, 
# it is recommended to enable offloading by specifying an offload directory.
OFFLOAD_DIR = None  # Specify your offload directory here, ensuring the directory exists.


def main():
    # device = torch.device(f"cuda:{GPU}")
    # text_embedder = T5Embedder(
    #     from_pretrained=MODEL_PATH,
    #     model_max_length=1024,
    #     device=device,
    #     use_offload_folder=OFFLOAD_DIR
    # )
    # tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model

    # tokens = tokenizer(
    #     INSTRUCTION, return_tensors="pt",
    #     padding="longest",
    #     truncation=True
    # )["input_ids"].to(device)
    #
    # tokens = tokens.view(1, -1)
    # with torch.no_grad():
    #     pred = text_encoder(tokens).last_hidden_state.detach().cpu()

    tker = Tokenizer()
    tker.process_task_instruction(TASK_NAME)

    # save_path = os.path.join(SAVE_DIR, f"{TASK_NAME}.pt")
    # # We save the embeddings in a dictionary format
    # torch.save({
    #     "name": TASK_NAME,
    #     "instruction": INSTRUCTION,
    #     "embeddings": pred
    # }, save_path
    # )
    #
    # print(
    #     f'\"{INSTRUCTION}\" from \"{TASK_NAME}\" is encoded by \"{MODEL_PATH}\" into shape {pred.shape} and saved to \"{save_path}\"')


class Tokenizer:

    def __init__(self):
        self.device = torch.device(f"cuda:0")
        text_embedder = T5Embedder(
            from_pretrained=MODEL_PATH,
            model_max_length=1024,
            device=self.device,
            use_offload_folder=OFFLOAD_DIR
        )
        self.tokenizer, self.text_encoder = text_embedder.tokenizer, text_embedder.model

    def process_task_instruction(self, task: str):
        instruction = INSTRUCTION[task]
        json_content = {}
        for k, txt in instruction.items():
            save_path = os.path.join(SAVE_DIR, task, f"{k}.pt")
            torch.save({
                "name": TASK_NAME,
                "instruction": INSTRUCTION,
                "embeddings": self.tokenize(txt)
            }, save_path)
            json_content[k] = save_path

        json_save_path = os.path.join(JSON_PATH, task, f"expanded_instruction_gpt-4-turbo.json.json")
        json.dump(json_content, open(json_save_path, "w"))
        print(instruction)
        print(json_content)
        print('save to json file: ', json_save_path)
        return json_content

    def tokenize(self, instruction: str):
        tokens = self.tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(self.device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_encoder(tokens).last_hidden_state.detach().cpu()
        return pred


# "put_puzzle/expanded_instruction_gpt-4-turbo.json"

if __name__ == "__main__":
    SAVE_DIR = "lang_embed/"
    JSON_PATH = "data/dataset/qkids/"
    if os.path.isdir(SAVE_DIR):
        main()
    else:
        print(f"Please create {SAVE_DIR} first.")
