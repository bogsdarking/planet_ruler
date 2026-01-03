Three approaches exist for identifying horizon pixels in an image: manual annotation by a human observer, 
automated detection based on image gradients, and semantic segmentation using machine learning. 
Each trades off precision, automation, and robustness differently.

Manual Annotation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A human observer clicks points along the visible horizon, providing ground truth with typical precision of 1-2 pixels. 
This method achieves the highest accuracy when the horizon is visually clear, as humans excel at pattern recognition 
and can distinguish true horizon from atmospheric features or clouds. The limitation is labor—annotating a single image 
requires 30-60 seconds and does not scale to batch processing.

Sparse computation optimization enables efficient model fitting from manual annotations. Rather than 
evaluating the cost function across the entire image, we compute predicted limb positions only at the 
annotated pixel x-coordinates, comparing predicted and observed y-values. This reduces computation 
from ~10 million pixel evaluations to ~100 point evaluations, achieving 85× speedup with no loss in accuracy.

Gradient-Based Detection  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Image gradients indicate transitions from sky to Earth, making gradient magnitude a natural horizon detector. 
The gradient-break method identifies the horizon as the y-coordinate of maximum vertical gradient magnitude 
for each image column. Savitzky-Golay smoothing reduces noise while preserving edge sharpness.

This approach works well for images with clean sky-to-ground transitions but struggles with atmospheric haze, 
cloud layers, or gradual transitions. The method is fully automatic and computationally efficient (typically 1-5 seconds), 
making it suitable for batch processing when scene complexity is low.

Segmentation-Based Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Segment Anything Model (SAM) provides semantic segmentation without task-specific training. Given an image, 
SAM generates masks corresponding to distinct objects or regions. The horizon appears as a strong boundary between 
sky and ground segments, allowing robust detection even in complex scenes with clouds, terrain features, or atmospheric layers.

SAM's computational cost is substantial (10-30 seconds on CPU, 2-5 seconds on GPU) but its robustness to scene 
complexity often justifies this expense. The model's ability to distinguish atmospheric features from the true 
planetary limb makes it particularly valuable for hazy or cloud-scattered images where gradient-based methods fail.

Method Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Precision**: Manual annotation > Segmentation ≥ Gradient-based

**Automation**: Gradient-based = Segmentation > Manual annotation

**Robustness to complexity**: Segmentation > Manual annotation > Gradient-based

**Computational cost**: Gradient-based < Manual annotation < Segmentation

For research requiring highest accuracy, manual annotation remains optimal. For automated pipelines processing 
clear horizons, gradient-based detection suffices. For robustness across varied scene complexity, segmentation 
provides the best balance despite higher computational cost.

Detection-Free Alternative
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The gradient-field method (detailed separately) bypasses explicit detection entirely by optimizing model parameters 
to align predicted limb position with image gradient vectors across the entire image. This approach uses all 
available image information rather than reducing it to a 1D horizon curve, trading computational cost for 
improved robustness and precision.