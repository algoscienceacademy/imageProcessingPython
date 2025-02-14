import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier, MLPRegressor
from scipy.ndimage import gaussian_filter, sobel, zoom, rotate
from sklearn.preprocessing import MinMaxScaler
import cv2
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass
from matplotlib.animation import FuncAnimation
import warnings
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
from pathlib import Path
import seaborn as sns

warnings.filterwarnings('ignore')

plt.ion()  # Enable interactive mode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NetworkConfig:
    """Neural network configuration parameters"""
    input_size: Tuple[int, int, int] = (28, 28, 3)
    hidden_layers: List[int] = (512, 256, 128, 64)
    learning_rate: float = 0.01
    max_iter: int = 200
    batch_size: int = 32

def create_sample_image(type='rgb'):
    size = 28
    if type == 'rgb':
        # Create more visible pattern
        image = np.zeros((size, size, 3))
        center = size // 2
        radius = size // 4
        
        # Add colorful circles
        for i, color in enumerate([
            [1, 0, 0],  # Red
            [0, 1, 0],  # Green
            [0, 0, 1]   # Blue
        ]):
            mask = np.zeros((size, size))
            y, x = np.ogrid[:size, :size]
            mask[(x - center)**2 + (y - center)**2 <= (radius + i*2)**2] = 1
            image[:,:,i] = mask * color[i]
        
        return image
    else:
        # Create a simple geometric pattern
        size = 28
        image = np.zeros((size, size))
        image[size//4:3*size//4, size//4:3*size//4] = 1
        return image

def create_image_dataset(n_samples=100):
    base_image = create_sample_image('rgb')
    X = []
    y = []
    
    for i in range(n_samples):
        angle = np.random.randint(0, 360)
        noise = np.random.normal(0, 0.1, base_image.shape)
        transformed = rotate(base_image, angle, reshape=False) + noise
        X.append(transformed.reshape(-1))
        y.append(angle // 90)
        
    return np.array(X), np.array(y)

def create_artistic_image(input_image, style='abstract'):
    """Generate artistic variations of input image"""
    img = input_image.copy()
    
    if style == 'abstract':
        # Apply multiple filters for abstract effect
        for i in range(3):
            img[:,:,i] = gaussian_filter(img[:,:,i], sigma=2)
            img[:,:,i] = sobel(img[:,:,i])
    elif style == 'painterly':
        # Create painterly effect
        img = cv2.stylization(img.astype(np.uint8), sigma_s=60, sigma_r=0.6)
    return img

class AdvancedImageGenerator:
    """Advanced image generation using neural networks"""
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.generator = self._build_generator()
        self.discriminator = self._build_discriminator()
        # Initialize generator with random weights
        random_data = np.random.randn(100, config.hidden_layers[0])
        random_target = np.random.randn(100, np.prod(config.input_size))
        self.generator.fit(random_data, random_target)
    
    def _build_generator(self) -> MLPRegressor:
        """Builds advanced generator network"""
        return MLPRegressor(
            hidden_layer_sizes=self.config.hidden_layers,
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iter,
            activation='relu',
            solver='adam',
            early_stopping=False,  # Disable early stopping
            validation_fraction=0.0  # Disable validation split
        )
    
    def _build_discriminator(self) -> MLPClassifier:
        """Builds discriminator network"""
        return MLPClassifier(
            hidden_layer_sizes=self.config.hidden_layers[::-1],
            learning_rate_init=self.config.learning_rate,
            max_iter=self.config.max_iter,
            activation='relu',
            solver='adam'
        )
    
    def generate_realistic_image(self, latent_vector: np.ndarray) -> np.ndarray:
        """Generates realistic images using advanced processing"""
        # Generate image directly without fitting
        base = self.generator.predict(latent_vector)
        base = base.reshape(self.config.input_size)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)
        
        # Apply enhancements
        enhanced = self._apply_realistic_effects(base)
        return self._enhance_details(enhanced)
    
    def _apply_realistic_effects(self, img: np.ndarray) -> np.ndarray:
        """Applies realistic effects using multiple techniques"""
        result = img.copy()
        
        # Apply multiple enhancements
        for i in range(3):
            # Enhance edges
            edges = sobel(result[:,:,i])
            # Smooth details
            smooth = gaussian_filter(result[:,:,i], sigma=1.5)
            # Combine effects
            result[:,:,i] = np.clip(smooth + 0.3 * edges, 0, 1)
        
        # Apply color correction
        result = np.power(result, 0.8)  # Gamma correction
        return result
    
    def _enhance_details(self, img: np.ndarray) -> np.ndarray:
        """Enhances image details using classical CV techniques"""
        # Convert to uint8 for CV operations
        img_uint8 = (img * 255).astype(np.uint8)
        
        # Apply series of enhancements
        blurred = cv2.GaussianBlur(img_uint8, (0, 0), 3)
        sharpened = cv2.addWeighted(img_uint8, 1.5, blurred, -0.5, 0)
        
        # Apply detail enhancement
        enhanced = cv2.detailEnhance(sharpened, sigma_s=10, sigma_r=0.15)
        
        return enhanced.astype(float) / 255.0

def create_complex_dataset(n_samples=100):
    patterns = []
    labels = []
    
    for i in range(n_samples):
        shape_type = i % 4
        img = np.zeros((28, 28))
        center = np.random.randint(10, 18, 2)
        
        if shape_type == 0:  # Circle
            rr, cc = np.ogrid[:28, :28]
            radius = np.random.randint(4, 8)
            img[((rr-center[0])**2 + (cc-center[1])**2) <= radius**2] = 1
        elif shape_type == 1:  # Square
            size = np.random.randint(6, 12)
            img[center[0]-size//2:center[0]+size//2,
                center[1]-size//2:center[1]+size//2] = 1
        elif shape_type == 2:  # Triangle
            from_center = np.random.randint(5, 10)
            points = np.array([
                [center[0]-from_center, center[1]-from_center],
                [center[0]-from_center, center[1]+from_center],
                [center[0]+from_center, center[1]]
            ])
            xx, yy = np.mgrid[:28, :28]
            img = np.zeros((28, 28), dtype=bool)
            for j in range(3):
                p1, p2 = points[j], points[(j+1)%3]
                mask = (yy-p1[1])*(p2[0]-p1[0]) > (xx-p1[0])*(p2[1]-p1[1])
                img = img | mask
            img = img.astype(float)
        else:  # Cross
            thickness = np.random.randint(2, 4)
            img[center[0]-8:center[0]+8, center[1]-thickness:center[1]+thickness] = 1
            img[center[0]-thickness:center[0]+thickness, center[1]-8:center[1]+8] = 1
        
        # Apply transformations
        angle = np.random.randint(0, 360)
        img = rotate(img, angle, reshape=False)
        img = gaussian_filter(img, sigma=0.5)
        img += np.random.normal(0, 0.1, img.shape)
        
        patterns.append(img.flatten())
        labels.append(shape_type)
    
    return np.array(patterns), np.array(labels)

def _plot_enhanced_network(mlp, ax, sample_idx):
    # Simplified network visualization
    layer_sizes = [mlp.coefs_[0].shape[0]] + [layer.shape[1] for layer in mlp.coefs_]
    
    # Plot nodes (reduced size and complexity)
    for i, size in enumerate(layer_sizes):
        x = np.full(size, i)
        y = np.linspace(0, 1, size)
        
        if i == 0:
            colors = plt.cm.viridis(np.linspace(0, 1, size))
        else:
            colors = plt.cm.viridis(np.random.rand(size))
        
        ax.scatter(x, y, c=colors, s=50)  # Reduced marker size
    
    # Plot only a subset of connections
    for i in range(len(mlp.coefs_)):
        weights = mlp.coefs_[i]
        # Plot only 10% of connections randomly
        mask = np.random.choice([True, False], size=weights.shape, p=[0.1, 0.9])
        for j, k in zip(*np.where(mask)):
            weight = weights[j, k]
            color = 'red' if weight > 0 else 'blue'
            ax.plot([i, i+1], 
                   [j/weights.shape[0], k/weights.shape[1]], 
                   c=color, alpha=0.1, linewidth=0.5)
    
    ax.set_title('Network Architecture')
    ax.set_xticks(range(len(layer_sizes)))
    ax.set_xticklabels(['In'] + [f'H{i+1}' for i in range(len(layer_sizes)-2)] + ['Out'])

# Main execution
if __name__ == '__main__':
    try:
        # Setup the display environment
        plt.style.use('default')  # Use default style for better visibility
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        
        # Create simple static visualization first
        logger.info("Creating visualization...")
        
        # Create dataset and network
        X, y = create_complex_dataset(n_samples=50)  # Reduced samples
        mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16),  # Even smaller network
            activation='relu',
            max_iter=1,
            warm_start=True,
            learning_rate_init=0.01
        )
        
        # Create main figure
        fig = plt.figure()
        plt.suptitle("Neural Network Training Visualization", fontsize=14)
        
        # Create 2x2 subplots
        ax1 = plt.subplot(221)
        ax2 = plt.subplot(222)
        ax3 = plt.subplot(223)
        ax4 = plt.subplot(224)
        
        # Initialize plots
        costs, accuracies = [], []
        
        # Training loop with manual updates
        logger.info("Starting training loop...")
        for i in range(50):  # Reduced iterations
            # Train one batch
            mlp.partial_fit(X, y, classes=np.unique(y))
            
            # Update metrics
            costs.append(mlp.loss_)
            accuracies.append(np.mean(mlp.predict(X) == y))
            
            # Clear and update plots
            for ax in [ax1, ax2, ax3, ax4]:
                ax.clear()
            
            # Plot loss
            ax1.plot(costs, 'r-')
            ax1.set_title('Training Loss')
            ax1.grid(True)
            
            # Plot accuracy
            ax2.plot(accuracies, 'g-')
            ax2.set_title('Accuracy')
            ax2.grid(True)
            
            # Plot sample
            ax3.imshow(X[i % len(X)].reshape(28, 28), cmap='viridis')
            ax3.set_title(f'Sample {i}')
            
            # Plot network structure
            _plot_enhanced_network(mlp, ax4, i % len(X))
            
            # Update display
            plt.tight_layout()
            plt.pause(0.1)  # Add small pause to see updates
            
            if i % 10 == 0:
                logger.info(f"Iteration {i}: Loss = {mlp.loss_:.4f}, Accuracy = {accuracies[-1]:.4f}")
        
        # Keep the final plot visible
        logger.info("Training complete. Close the window to exit.")
        plt.show(block=True)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    finally:
        plt.close('all')
        logger.info("Process completed")