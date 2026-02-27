"""
Image Captioning AI System
Combines VGG16/ResNet for feature extraction with LSTM for caption generation
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Embedding, Dropout, 
                                     Add, Concatenate, Bidirectional)
import pickle
import os
from tqdm import tqdm

class ImageCaptioningModel:
    """
    A complete image captioning system that uses transfer learning
    from pre-trained CNNs and generates captions using LSTM/Transformer
    """
    
    def __init__(self, max_caption_length=40, vocab_size=10000, 
                 embedding_dim=256, cnn_model='vgg16'):
        """
        Initialize the image captioning model
        
        Args:
            max_caption_length: Maximum length of generated captions
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            cnn_model: 'vgg16' or 'resnet50'
        """
        self.max_length = max_caption_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.cnn_model_name = cnn_model
        
        # Initialize feature extractor
        self.feature_extractor = self._build_feature_extractor()
        
        # Placeholders for tokenizer and model
        self.tokenizer = None
        self.model = None
        self.feature_dim = 4096 if cnn_model == 'vgg16' else 2048
        
    def _build_feature_extractor(self):
        """Build CNN feature extractor using pre-trained models"""
        if self.cnn_model_name == 'vgg16':
            base_model = VGG16(weights='imagenet', include_top=True)
            # Extract features from fc2 layer (4096 dimensions)
            model = Model(inputs=base_model.input, 
                         outputs=base_model.layers[-2].output)
        else:  # resnet50
            base_model = ResNet50(weights='imagenet', include_top=True)
            # Extract features from avg_pool layer (2048 dimensions)
            model = Model(inputs=base_model.input, 
                         outputs=base_model.layers[-2].output)
        
        return model
    
    def extract_features(self, image_path):
        """
        Extract features from an image using pre-trained CNN
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        try:
            # Load and preprocess image
            img = load_img(image_path, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Preprocess based on model type
            if self.cnn_model_name == 'vgg16':
                from tensorflow.keras.applications.vgg16 import preprocess_input
            else:
                from tensorflow.keras.applications.resnet50 import preprocess_input
            
            img_array = preprocess_input(img_array)
            
            # Extract features
            features = self.feature_extractor.predict(img_array, verbose=0)
            return features.flatten()
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return None
    
    def build_caption_model(self):
        """
        Build the caption generation model using LSTM
        Architecture: Image features + Word embeddings -> LSTM -> Dense
        """
        # Image feature input
        image_input = Input(shape=(self.feature_dim,))
        image_dense = Dropout(0.5)(image_input)
        image_dense = Dense(256, activation='relu')(image_dense)
        
        # Caption sequence input
        caption_input = Input(shape=(self.max_length,))
        caption_embed = Embedding(self.vocab_size, self.embedding_dim, 
                                  mask_zero=True)(caption_input)
        caption_embed = Dropout(0.5)(caption_embed)
        
        # LSTM for sequence processing
        caption_lstm = LSTM(256, return_sequences=False)(caption_embed)
        
        # Merge image and caption features
        merged = Add()([image_dense, caption_lstm])
        merged = Dense(256, activation='relu')(merged)
        merged = Dropout(0.5)(merged)
        
        # Output layer
        output = Dense(self.vocab_size, activation='softmax')(merged)
        
        # Create model
        model = Model(inputs=[image_input, caption_input], outputs=output)
        model.compile(loss='categorical_crossentropy', 
                     optimizer='adam',
                     metrics=['accuracy'])
        
        self.model = model
        return model
    
    def create_tokenizer(self, captions):
        """
        Create and fit tokenizer on caption data
        
        Args:
            captions: List of caption strings
        """
        from tensorflow.keras.preprocessing.text import Tokenizer
        
        tokenizer = Tokenizer(num_words=self.vocab_size, 
                            oov_token="<unk>",
                            filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(captions)
        self.tokenizer = tokenizer
        
        # Add special tokens
        self.word_to_idx = tokenizer.word_index
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.idx_to_word[0] = '<pad>'
        
        return tokenizer
    
    def prepare_training_data(self, image_features, captions):
        """
        Prepare training data for the model
        
        Args:
            image_features: Dictionary mapping image_id to feature vector
            captions: Dictionary mapping image_id to list of captions
            
        Returns:
            X_img, X_cap, y arrays for training
        """
        X_img, X_cap, y = [], [], []
        
        for img_id, caption_list in tqdm(captions.items(), desc="Preparing data"):
            if img_id not in image_features:
                continue
                
            feature = image_features[img_id]
            
            for caption in caption_list:
                # Tokenize caption
                seq = self.tokenizer.texts_to_sequences([caption])[0]
                
                # Create multiple training samples from one caption
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]
                    
                    # Pad input sequence
                    in_seq = pad_sequences([in_seq], 
                                          maxlen=self.max_length,
                                          padding='post')[0]
                    
                    # One-hot encode output
                    out_seq_encoded = np.zeros(self.vocab_size)
                    if out_seq < self.vocab_size:
                        out_seq_encoded[out_seq] = 1
                    
                    X_img.append(feature)
                    X_cap.append(in_seq)
                    y.append(out_seq_encoded)
        
        return np.array(X_img), np.array(X_cap), np.array(y)
    
    def train(self, X_img, X_cap, y, epochs=20, batch_size=32, validation_split=0.2):
        """Train the caption model"""
        if self.model is None:
            self.build_caption_model()
        
        # Callbacks
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                   patience=5,
                                                   restore_best_weights=True)
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                      factor=0.5,
                                                      patience=3,
                                                      min_lr=1e-6)
        
        history = self.model.fit(
            [X_img, X_cap], y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        return history
    
    def generate_caption(self, image_path, max_words=40, beam_width=3):
        """
        Generate caption for an image using beam search
        
        Args:
            image_path: Path to image file
            max_words: Maximum number of words in caption
            beam_width: Beam width for beam search
            
        Returns:
            Generated caption string
        """
        # Extract features
        feature = self.extract_features(image_path)
        if feature is None:
            return "Error processing image"
        
        feature = feature.reshape(1, -1)
        
        # Start with empty sequence
        sequences = [[[], 0.0]]
        
        for _ in range(max_words):
            all_candidates = []
            
            for seq, score in sequences:
                # Pad sequence
                padded_seq = pad_sequences([seq], 
                                          maxlen=self.max_length,
                                          padding='post')
                
                # Predict next word probabilities
                preds = self.model.predict([feature, padded_seq], verbose=0)[0]
                
                # Get top beam_width predictions
                top_indices = np.argsort(preds)[-beam_width:]
                
                for idx in top_indices:
                    candidate = [seq + [idx], score + np.log(preds[idx])]
                    all_candidates.append(candidate)
            
            # Select top beam_width sequences
            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            
            # Stop if all sequences end with 0 (padding)
            if all([seq[-1] == 0 if len(seq) > 0 else False for seq, _ in sequences]):
                break
        
        # Get best sequence
        best_seq = sequences[0][0]
        
        # Convert to words
        caption_words = []
        for idx in best_seq:
            if idx == 0 or idx not in self.idx_to_word:
                continue
            word = self.idx_to_word[idx]
            if word == '<unk>':
                continue
            caption_words.append(word)
        
        caption = ' '.join(caption_words)
        return caption.capitalize()
    
    def save_model(self, model_path='image_caption_model.h5', 
                   tokenizer_path='tokenizer.pkl'):
        """Save model and tokenizer"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
        
        if self.tokenizer:
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(self.tokenizer, f)
            print(f"Tokenizer saved to {tokenizer_path}")
    
    def load_model(self, model_path='image_caption_model.h5',
                   tokenizer_path='tokenizer.pkl'):
        """Load model and tokenizer"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
        
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        self.word_to_idx = self.tokenizer.word_index
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.idx_to_word[0] = '<pad>'
        print(f"Tokenizer loaded from {tokenizer_path}")


# Example usage and demonstration
def demo_training():
    """
    Demonstration of how to train the model
    You would need to provide your own dataset
    """
    print("=== Image Captioning AI Demo ===\n")
    
    # Initialize model
    caption_model = ImageCaptioningModel(
        max_caption_length=40,
        vocab_size=10000,
        embedding_dim=256,
        cnn_model='vgg16'  # or 'resnet50'
    )
    
    # Example: Extract features from images
    # In real scenario, you would have a dataset like Flickr8k or MS COCO
    print("Step 1: Extract features from images")
    image_features = {}
    
    # Example images (replace with your dataset)
    # for img_id, img_path in your_image_dataset.items():
    #     features = caption_model.extract_features(img_path)
    #     image_features[img_id] = features
    
    # Example: Create captions dictionary
    captions = {
        # 'image1': ['a dog playing in the park', 'brown dog running outside'],
        # 'image2': ['a cat sitting on a chair', 'grey cat resting indoors'],
    }
    
    print("\nStep 2: Create tokenizer from captions")
    all_captions = [cap for cap_list in captions.values() for cap in cap_list]
    caption_model.create_tokenizer(all_captions)
    
    print("\nStep 3: Prepare training data")
    # X_img, X_cap, y = caption_model.prepare_training_data(image_features, captions)
    
    print("\nStep 4: Build and train model")
    caption_model.build_caption_model()
    # history = caption_model.train(X_img, X_cap, y, epochs=20, batch_size=32)
    
    print("\nStep 5: Generate captions for new images")
    # caption = caption_model.generate_caption('path/to/new/image.jpg')
    # print(f"Generated caption: {caption}")
    
    print("\nStep 6: Save trained model")
    # caption_model.save_model()
    
    print("\n=== Demo Complete ===")
    print("\nTo use this code:")
    print("1. Prepare your dataset (Flickr8k, MS COCO, or custom)")
    print("2. Extract features for all images")
    print("3. Train the model with your captions")
    print("4. Generate captions for new images")
    print("\nDataset recommendations:")
    print("- Flickr8k: Good for learning (8k images)")
    print("- Flickr30k: Medium size (30k images)")
    print("- MS COCO: Large scale (120k images)")


if __name__ == "__main__":
    demo_training()
