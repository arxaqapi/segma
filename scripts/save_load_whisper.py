from transformers import WhisperModel

if __name__ == "__main__":
    model = WhisperModel.from_pretrained("openai/whisper-tiny")
    model.encoder.save_pretrained("whisper_tiny_encoder")
