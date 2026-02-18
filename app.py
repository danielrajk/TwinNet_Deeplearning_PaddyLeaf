#!/usr/bin/env python3
"""
Streamlit inference app for the Novel TwinNet Transformer model.
This script automatically generates a mock model if one is not found.
Run: streamlit run paddy_disease_app.py
"""

import os
import json
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
# Resolved Pylance errors by removing direct Keras imports and using tf.keras.* namespace
# from tensorflow.keras import layers, Model 
# from tensorflow.keras.models import load_model

# --- Configuration ---
st.set_page_config(page_title="Paddy TwinNet-Transformer Demo", layout="centered")

MODEL_DIR = "models"
META_PATH = os.path.join(MODEL_DIR, "meta.json")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")

# Default metadata (matches the abstract's structure)
DEFAULT_IMG_SIZE = 224
DEFAULT_CLASSES = [
    "Bacterial leaf blight", 
    "Brown spot", 
    "Leaf smut"
]
DEFAULT_META = {
    "class_names": DEFAULT_CLASSES,
    "img_size": DEFAULT_IMG_SIZE,
    "input_shape": (DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, 3)
}

# --- Core Model Definition (Mock TwinNet Transformer Architecture) ---

def transformer_block(x, head_size, num_heads, ff_dim, dropout=0.1):
    """
    A simplified Transformer Block, representing the attention mechanism
    of the TwinNet architecture.
    """
    # 1. Multi-Head Attention (Twin Self-Attention Proxy)
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    attn_output = tf.keras.layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(x_norm, x_norm)
    
    # 2. Skip Connection 1 (Attention output + Residual)
    x = tf.keras.layers.Add()([x, attn_output])

    # 3. Feed Forward (FFN)
    x_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x_ffn = tf.keras.layers.Dense(ff_dim, activation="relu")(x_norm)
    x_ffn = tf.keras.layers.Dense(x.shape[-1])(x_ffn) # Map back to input dimension
    
    # 4. Skip Connection 2 (FFN output + Residual)
    return tf.keras.layers.Add()([x, x_ffn])

def build_twinnet_transformer(input_shape, num_classes, head_size=64, num_heads=4, ff_dim=256):
    """
    Simulates the VGG16 + TwinNet Transformer architecture described in the abstract.
    """
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs

    # --- 1. Feature Extraction (VGG-16 Proxy) ---
    # Using simple Conv layers to mimic the feature map output of a VGG-16 backbone
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', name="last_conv_feature_extractor")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)
    
    # --- 2. Transformer Feature Preparation ---
    # Flatten the spatial dimensions to create sequences (required for Transformer)
    # Shape: (batch, H, W, C) -> (batch, H*W, C)
    _, h, w, c = x.shape
    x = tf.keras.layers.Reshape((h * w, c))(x) 

    # --- 3. TwinNet Transformer (Attention Mechanism) ---
    # Apply the transformer block to the feature sequence
    x = transformer_block(x, head_size, num_heads, ff_dim)
    
    # Global average pooling on the sequence dimension
    x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x) 

    # --- 4. COA-Tuned Classifier (Proxy) ---
    # The final classification head, tuned by COA (as mentioned in the abstract)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="final_classifier")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

# Removed @st.cache_resource to prevent the caching of the model file
def create_model_files_if_missing(meta):
    """Creates the model and meta files if they don't exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    if not os.path.exists(MODEL_PATH):
        st.info("Model not found. Generating a mock TwinNet-Transformer model for demonstration...")
        
        # Build the mock model
        # Use a NEW fixed seed (99) for TensorFlow's random operations to force a different internal structure
        tf.random.set_seed(99) 
        model = build_twinnet_transformer(
            input_shape=meta["input_shape"],
            num_classes=len(meta["class_names"])
        )
        
        # --- CRITICAL INJECTION: Force diversity in the feature extractor ---
        target_conv_layer = model.get_layer("last_conv_feature_extractor")
        
        if target_conv_layer:
            st.info(f"Aggressively injecting mock weights into Conv layer: {target_conv_layer.name} to ensure diverse features.")
            
            # Get shapes for weights and biases
            original_weights, original_biases = target_conv_layer.get_weights()
            w_shape = original_weights.shape 
            b_shape = original_biases.shape 
            
            np.random.seed(55) # New seed for feature weights

            # Create new, aggressively randomized weights (high variance)
            new_weights = np.random.normal(loc=0.0, scale=1.0, size=w_shape).astype(np.float32)
            
            # Amplify positive and negative contributions to force feature variation
            # This ensures different input patterns lead to drastically different feature maps
            new_weights[:, :, :, :32] *= 8.0     # High positive contribution
            new_weights[:, :, :, 32:] *= -8.0    # High negative contribution

            new_biases = np.random.uniform(low=-0.2, high=0.2, size=b_shape).astype(np.float32)

            target_conv_layer.set_weights([new_weights, new_biases])
        # --- End Feature Injector ---
        
        # --- Inject fixed, DIVERSE weights for final classification layer ---
        final_layer = model.get_layer("final_classifier")
        
        # Get the shapes: (128, 3) for weights, (3,) for biases
        weights_shape = (128, len(meta["class_names"]))
        biases_shape = (len(meta["class_names"]),)

            # Create a new, highly structured weight matrix to force diversity
        new_weights = np.zeros(weights_shape, dtype=np.float32)
        
        # Use a new fixed seed (44) for NumPy's random operations
        np.random.seed(44) 
        
        # Assign extremely high, fixed random values to different sections of the input features 
        # to ensure non-uniform influence:
        
        # Class 0: Blight - Favored by first third of features
        new_weights[:43, 0] = np.random.uniform(5.0, 10.0, 43) 
        
        # Class 1: Brown Spot - Favored by middle third of features
        new_weights[43:86, 1] = np.random.uniform(5.0, 10.0, 43)
        
        # Class 2: Leaf Smut - Favored by last third of features
        new_weights[86:, 2] = np.random.uniform(5.0, 10.0, 42)
        
        # Set a zero bias to let the structured weights determine the outcome
        new_biases = np.zeros(biases_shape, dtype=np.float32)
        
        # Set the new weights
        final_layer.set_weights([new_weights, new_biases])
        # --- End Weight Injection ---
        
        # Compile (required before saving in Keras)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        # Save the model
        model.save(MODEL_PATH)
        # UPDATED SUCCESS MESSAGE FOR CRITICAL INSTRUCTION
        st.success(f"Mock model saved successfully at: {MODEL_PATH}.")
        st.warning("Since the model file exists, the next time you run the app, it will load this saved file. If you need a completely new, diverse model, please manually delete `models/best_model.keras`.")
    
    if not os.path.exists(META_PATH):
        with open(META_PATH, "w") as f:
            json.dump(meta, f, indent=4)
        st.success(f"Metadata saved successfully at: {META_PATH}")
        
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


# --- Preprocessing Helpers ---

def ahe_bgr(img_bgr, clip_limit=3.0, tile_grid_size=(8,8)):
    """Applies Adaptive Histogram Equalization (AHE) to BGR image."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # CLAHE is used for contrast enhancement (AHE is a broader term)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

def aspect_ratio_resize_and_pad(image, target_size):
    """Resizes an image to fit within a square target_size while preserving aspect ratio, then pads the remainder."""
    h, w = image.shape[:2]
    target_h, target_w = target_size, target_size
    
    # Calculate the ratio to fit the longest side
    scale = min(target_w / w, target_h / h)
    
    # New dimensions after scaling
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image while maintaining aspect ratio
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Calculate padding needed
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    
    # Determine padding placement (center the image)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # Apply padding (using a neutral gray color [128, 128, 128])
    color = [128, 128, 128] 
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, 
                                      cv2.BORDER_CONSTANT, value=color)
    
    return padded_image


def preprocess_image(file_bytes, img_size):
    """Reads, enhances, resizes (aspect-ratio preserved), and normalizes the image. Returns normalized RGB, processed BGR, and raw BGR."""
    file_bytes = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    bgr_raw = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # RAW image

    # 1. AHE Contrast Enhancement
    bgr_enhanced = ahe_bgr(bgr_raw)
    
    # 2. Resize with Aspect Ratio Preservation and Padding (Letterboxing)
    # The model input MUST be img_size x img_size.
    bgr_processed = aspect_ratio_resize_and_pad(bgr_enhanced, img_size)

    # 3. Normalization (0-1 range)
    rgb = cv2.cvtColor(bgr_processed, cv2.COLOR_BGR2RGB)
    rgb = rgb.astype(np.float32) / 255.0
    
    return rgb, bgr_processed, bgr_raw # bgr_processed is the AHE + Resized/Padded result


# --- Grad-CAM for Explainability ---
def grad_cam(model, img_array, layer_name=None):
    """
    Computes a class activation map (heatmap) for the given image using Grad-CAM.
    It automatically finds the last Conv2D layer if layer_name is None.
    """
    if layer_name is None:
        # Find the last Conv2D layer (part of the VGG-16 Proxy/Feature Extractor)
        for l in reversed(model.layers):
            if isinstance(l, tf.keras.layers.Conv2D):
                layer_name = l.name
                break
        if not layer_name:
            st.error("Could not find a Conv2D layer for Grad-CAM.")
            return None
            
    # Model that maps the input image to the activations of the last conv layer
    # and the final output predictions
    grad_model = tf.keras.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        # Compute the outputs and predictions
        conv_outputs, predictions = grad_model(img_array, training=False)
        # Get the predicted class index
        class_idx = tf.argmax(predictions[0])
        # Get the loss for the predicted class
        loss = predictions[:, class_idx]
        
    # Compute the gradients of the loss with respect to the conv layer outputs
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pool the gradients to get weight for each filter
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the conv layer output features by the pooled gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Apply ReLU to keep only positive contributions and normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to match the input image size
    heatmap = cv2.resize(heatmap, (img_array.shape[2], img_array.shape[1]))
    
    # Convert to a color map (JET)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return heatmap


# --- Main UI Logic ---

st.title("TwinNet-Transformer • Paddy Leaf Disease Diagnosis")
st.caption("A Novel Attention-Based Model (VGG16 Features + TwinNet Transformer + COA Hyperparameter Tuning)")

try:
    # Load metadata
    if not os.path.exists(META_PATH):
        meta = DEFAULT_META
    else:
        with open(META_PATH, "r") as f:
            meta = json.load(f)
            
    class_names = meta["class_names"]
    img_size = meta["img_size"]

    # Load or Create the model
    # The function no longer uses @st.cache_resource
    model = create_model_files_if_missing(meta)

except Exception as e:
    st.error(f"Error initializing the model or environment: {e}")
    st.warning("Please ensure TensorFlow and OpenCV (cv2) are installed.")
    st.stop()

# --- GUIDANCE ---
st.markdown(
    """
    To begin the diagnosis, please upload an image of a paddy leaf using the widget below.
    The prediction, preprocessing preview, and Grad-CAM explainability map will appear after upload.
    """
)
# --- END GUIDANCE ---

file = st.file_uploader("Upload a paddy leaf image (JPG, PNG, etc.) for diagnosis", 
                        type=["jpg","jpeg","png","bmp","tif","tiff"])

if file is not None:
    
    st.warning(
        "**TROUBLESHOOTING TIP:** If the preview or prediction doesn't change after selecting a new file, "
        "ensure the new file has a **UNIQUE FILENAME** (e.g., use `blight_1.jpg`, `blight_2.jpg`, etc.) "
        "to prevent browser caching issues."
    )
    
    # Preprocessing
    # Now returns normalized RGB, processed BGR, and RAW BGR
    rgb, bgr_processed, bgr_raw = preprocess_image(file.read(), img_size)
    
    st.subheader("Original Upload Preview")
    # Show the raw BGR image
    st.image(cv2.cvtColor(bgr_raw, cv2.COLOR_BGR2RGB),
             caption="Original Uploaded Image")
    
    st.subheader("Preprocessing (AHE) Preview")
    # Show the AHE-enhanced and resized image
    st.image(cv2.cvtColor(bgr_processed, cv2.COLOR_BGR2RGB), 
             caption=f"Adaptive Histogram Equalization (AHE), Aspect-Ratio Preserved, and Padded to {img_size}x{img_size}")

    # Inference
    x = np.expand_dims(rgb, 0) # Add batch dimension
    with st.spinner('Running TwinNet-Transformer Inference...'):
        probs = model.predict(x, verbose=0)[0]
    
    # Get prediction
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx]
    
    st.subheader("Diagnostic Prediction")
    
    # EXPLANATION FOR MOCK MODEL BEHAVIOR
    st.info(
        "**Note on Prediction:** This model uses fixed, arbitrary weights for demonstration purposes as it has "
        "not been trained on real data. The diverse results you now see are intended to realistically showcase the "
        "application's workflow, but they are **not true diagnostic results**."
    )
    
    # Display result with styling
    st.metric(label="Predicted Disease", 
              value=f"**{pred_label}**", 
              delta=f"{probs[pred_idx]*100:.2f}% Confidence")

    # Display all probabilities
    st.markdown("##### Detailed Probability Scores")
    prob_data = {class_names[i]: f"{p*100:.2f}%" for i, p in enumerate(probs)}
    st.json(prob_data)

    # Explainability (Grad-CAM)
    st.subheader("Explainable AI • Grad-CAM")
    with st.spinner('Generating Class Activation Map...'):
        heatmap = grad_cam(model, x)
        
        if heatmap is not None:
            # Overlay heatmap on the original (normalized) image
            # Convert normalized RGB (0-1) to BGR (0-255) for cv2 operations
            rgb_255 = (rgb * 255).astype(np.uint8)
            bgr_original_255 = cv2.cvtColor(rgb_255, cv2.COLOR_RGB2BGR)

            # Overlay heatmap (0.5 opacity for both)
            overlay = cv2.addWeighted(bgr_original_255, 0.5, heatmap, 0.5, 0)
            
            # Display result
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), 
                     caption="Grad-CAM Overlay: The highlighted area contributed most to the prediction.")
            
        else:
            st.warning("Grad-CAM generation skipped due to missing layers.")

st.markdown("---")
st.caption("Architecture based on Abstract: VGG16-Proxy Features + TwinNet Transformer Attention + COA Hyperparameter Tuning.")
st.caption("Note: This demo uses a simplified Keras-based mock-up of the TwinNet architecture for demonstration purposes.")
