# Copyright 2025 Brandon Anderson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from planet_ruler.image import (
    MaskSegmenter,
    SegmentationBackend,
    CustomBackend,
    SAMBackend,
)


class TestSegmentationBackend:
    """Test base SegmentationBackend class."""
    
    def test_base_backend_not_implemented(self):
        """Base backend should raise NotImplementedError."""
        backend = SegmentationBackend()
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(NotImplementedError):
            backend.segment(image)


class TestCustomBackend:
    """Test CustomBackend with user-provided segmentation functions."""
    
    def test_custom_backend_with_ndarray_masks(self):
        """Test custom backend with numpy array masks."""
        # Define simple custom segmentation function
        def simple_segmenter(image):
            h, w = image.shape[:2]
            # Return two masks: upper half and lower half
            mask1 = np.zeros((h, w), dtype=bool)
            mask1[:h//2, :] = True
            mask2 = np.zeros((h, w), dtype=bool)
            mask2[h//2:, :] = True
            return [mask1, mask2]
        
        backend = CustomBackend(simple_segmenter)
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        masks = backend.segment(image)
        
        assert len(masks) == 2
        assert all('mask' in m for m in masks)
        assert all('area' in m for m in masks)
        assert all('id' in m for m in masks)
        assert masks[0]['area'] == 5000  # 50 rows * 100 cols
        assert masks[1]['area'] == 5000
    
    def test_custom_backend_with_dict_masks(self):
        """Test custom backend with dictionary masks."""
        def dict_segmenter(image):
            h, w = image.shape[:2]
            mask = np.ones((h, w), dtype=bool)
            return [{'mask': mask, 'area': h*w, 'score': 0.95}]
        
        backend = CustomBackend(dict_segmenter)
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        masks = backend.segment(image)
        
        assert len(masks) == 1
        assert 'mask' in masks[0]
        assert 'score' in masks[0]  # Custom field preserved
        assert masks[0]['area'] == 2500
    
    def test_custom_backend_invalid_mask_type(self):
        """Test custom backend with invalid mask type raises error."""
        def bad_segmenter(image):
            return ["not", "a", "mask"]  # Strings, not masks
        
        backend = CustomBackend(bad_segmenter)
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        with pytest.raises(TypeError, match="Mask must be ndarray or dict"):
            backend.segment(image)


class TestSAMBackend:
    """Test SAM backend with mocking."""
    
    @patch('planet_ruler.image.HAS_SEGMENT_ANYTHING', True)
    @patch('planet_ruler.image.kagglehub.model_download')
    @patch('planet_ruler.image.sam_model_registry')
    @patch('planet_ruler.image.SamAutomaticMaskGenerator')
    def test_sam_backend_init(self, mock_generator, mock_registry, mock_download):
        """Test SAM backend initialization."""
        mock_download.return_value = "/fake/path/to/model"
        
        backend = SAMBackend(model_size="vit_b")
        
        assert backend.model_size == "vit_b"
        assert backend.model_path == "/fake/path/to/model"
        mock_download.assert_called_once_with(
            "metaresearch/segment-anything/pyTorch/vit-b"
        )
    
    @patch('planet_ruler.image.HAS_SEGMENT_ANYTHING', False)
    def test_sam_backend_missing_dependencies(self):
        """Test SAM backend raises error when dependencies missing."""
        with pytest.raises(ImportError, match="segment-anything dependencies not available"):
            SAMBackend()
    
    @patch('planet_ruler.image.HAS_SEGMENT_ANYTHING', True)
    @patch('planet_ruler.image.kagglehub.model_download')
    @patch('planet_ruler.image.sam_model_registry')
    @patch('planet_ruler.image.SamAutomaticMaskGenerator')
    def test_sam_backend_segment(self, mock_generator_class, mock_registry, mock_download):
        """Test SAM backend segmentation."""
        mock_download.return_value = "/fake/path"
        
        # Mock SAM outputs
        mock_sam_instance = Mock()
        mock_registry.return_value = mock_sam_instance
        
        mock_generator = Mock()
        mock_generator_class.return_value = mock_generator
        
        # SAM returns masks with 'segmentation' key
        fake_masks = [
            {'segmentation': np.ones((50, 50), dtype=bool), 'area': 2500},
            {'segmentation': np.zeros((50, 50), dtype=bool), 'area': 0}
        ]
        mock_generator.generate.return_value = fake_masks
        
        backend = SAMBackend(model_size="vit_b")
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        masks = backend.segment(image)
        
        assert len(masks) == 2
        mock_generator.generate.assert_called_once_with(image)
    
    @patch('planet_ruler.image.HAS_SEGMENT_ANYTHING', True)
    @patch('planet_ruler.image.kagglehub.model_download')
    def test_sam_backend_underscore_to_hyphen(self, mock_download):
        """Test SAM backend converts underscores to hyphens for model names."""
        mock_download.return_value = "/fake/path"
        
        backend = SAMBackend(model_size="vit_h")
        
        # Should call with hyphen
        mock_download.assert_called_once_with(
            "metaresearch/segment-anything/pyTorch/vit-h"
        )


class TestMaskSegmenterInit:
    """Test MaskSegmenter initialization and backend creation."""
    
    def test_init_with_custom_backend(self):
        """Test initialization with custom backend."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def dummy_segmenter(img):
            return []
        
        segmenter = MaskSegmenter(
            image,
            method="custom",
            segment_fn=dummy_segmenter,
            interactive=False
        )
        
        assert segmenter.image.shape == (100, 100, 3)
        assert segmenter.original_shape == (100, 100)
        assert segmenter.downsample_factor == 1
        assert segmenter.interactive is False
        assert isinstance(segmenter._backend, CustomBackend)
    
    def test_init_custom_without_function_raises_error(self):
        """Test custom method without segment_fn raises error."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="custom method requires 'segment_fn'"):
            MaskSegmenter(image, method="custom")
    
    def test_init_unknown_method_raises_error(self):
        """Test unknown method raises error."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        with pytest.raises(ValueError, match="Unknown segmentation method"):
            MaskSegmenter(image, method="nonexistent_method")
    
    def test_init_with_downsampling(self):
        """Test initialization with downsampling factor."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        def dummy_segmenter(img):
            return []
        
        segmenter = MaskSegmenter(
            image,
            method="custom",
            segment_fn=dummy_segmenter,
            downsample_factor=4
        )
        
        assert segmenter.downsample_factor == 4


class TestClassifyAutomatic:
    """Test automatic mask classification heuristic."""
    
    def test_classify_automatic_basic(self):
        """Test automatic classification with simple upper/lower masks."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def simple_segmenter(img):
            h, w = img.shape[:2]
            # Upper mask (sky)
            mask1 = np.zeros((h, w), dtype=bool)
            mask1[:30, :] = True
            # Lower mask (planet)
            mask2 = np.zeros((h, w), dtype=bool)
            mask2[70:, :] = True
            return [
                {'mask': mask1, 'area': 3000},
                {'mask': mask2, 'area': 3000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=simple_segmenter, interactive=False
        )
        segmenter._masks = simple_segmenter(image)
        
        classified = segmenter._classify_automatic()
        
        assert len(classified['sky']) == 1
        assert len(classified['planet']) == 1
        assert len(classified['exclude']) == 0
        
        # Upper mask should be sky
        assert classified['sky'][0]['mask'][0, 0] == True
        # Lower mask should be planet
        assert classified['planet'][0]['mask'][90, 0] == True
    
    def test_classify_automatic_picks_largest(self):
        """Test that automatic classification picks largest mask in each half."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def multi_mask_segmenter(img):
            h, w = img.shape[:2]
            # Two upper masks (different sizes)
            mask1 = np.zeros((h, w), dtype=bool)
            mask1[:10, :] = True  # Small
            mask2 = np.zeros((h, w), dtype=bool)
            mask2[10:40, :] = True  # Large
            # Two lower masks
            mask3 = np.zeros((h, w), dtype=bool)
            mask3[60:70, :] = True  # Small
            mask4 = np.zeros((h, w), dtype=bool)
            mask4[70:, :] = True  # Large
            
            return [
                {'mask': mask1, 'area': 1000},
                {'mask': mask2, 'area': 3000},  # Largest upper
                {'mask': mask3, 'area': 1000},
                {'mask': mask4, 'area': 3000}   # Largest lower
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=multi_mask_segmenter, interactive=False
        )
        segmenter._masks = multi_mask_segmenter(image)
        
        classified = segmenter._classify_automatic()
        
        # Should pick largest in each half
        assert classified['sky'][0]['area'] == 3000
        assert classified['planet'][0]['area'] == 3000
    
    def test_classify_automatic_too_few_masks(self):
        """Test error when too few masks for automatic classification."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def single_mask_segmenter(img):
            return [{'mask': np.ones((100, 100), dtype=bool), 'area': 10000}]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=single_mask_segmenter, interactive=False
        )
        segmenter._masks = single_mask_segmenter(image)
        
        with pytest.raises(ValueError, match="Need at least 2 masks"):
            segmenter._classify_automatic()


class TestCombineMasks:
    """Test limb extraction from classified masks."""
    
    def test_combine_masks_simple_horizon(self):
        """Test limb extraction with simple sky/planet separation."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def horizon_segmenter(img):
            h, w = img.shape[:2]
            # Sky: upper 40 rows
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[:40, :] = True
            # Planet: lower 60 rows
            planet_mask = np.zeros((h, w), dtype=bool)
            planet_mask[60:, :] = True
            
            return [
                {'mask': sky_mask, 'area': 4000},
                {'mask': planet_mask, 'area': 6000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=horizon_segmenter, interactive=False
        )
        segmenter._masks = horizon_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Limb should be at row 49.5
        # Sky ends at row 39 (last True), planet starts at row 60 (first True)
        # Average: (39 + 60) / 2 = 49.5
        assert len(limb) == 100  # One y-value per column
        assert np.allclose(limb, 49.5)
    
    def test_combine_masks_with_gap(self):
        """Test limb extraction with gap between sky and planet."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def gap_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[:30, :] = True
            planet_mask = np.zeros((h, w), dtype=bool)
            planet_mask[70:, :] = True
            
            return [
                {'mask': sky_mask, 'area': 3000},
                {'mask': planet_mask, 'area': 3000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=gap_segmenter, interactive=False
        )
        segmenter._masks = gap_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Sky ends at row 29, planet starts at row 70
        # Average: (29 + 70) / 2 = 49.5
        assert len(limb) == 100
        assert np.allclose(limb, 49.5)
    
    def test_combine_masks_overlapping(self):
        """Test limb extraction with overlapping masks."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def overlap_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            sky_mask[:60, :] = True  # Extends to row 60
            planet_mask = np.zeros((h, w), dtype=bool)
            planet_mask[40:, :] = True  # Starts at row 40
            
            return [
                {'mask': sky_mask, 'area': 6000},
                {'mask': planet_mask, 'area': 6000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=overlap_segmenter, interactive=False
        )
        segmenter._masks = overlap_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Sky ends at row 59, planet starts at row 40
        # Average: (59 + 40) / 2 = 49.5
        assert len(limb) == 100
        assert np.allclose(limb, 49.5)
    
    def test_combine_masks_only_planet(self):
        """Test limb extraction when only planet labeled."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def planet_only_segmenter(img):
            h, w = img.shape[:2]
            planet_mask = np.zeros((h, w), dtype=bool)
            planet_mask[60:, :] = True
            
            return [{'mask': planet_mask, 'area': 4000}]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=planet_only_segmenter, interactive=False
        )
        segmenter._masks = planet_only_segmenter(image)
        
        classified = {
            'planet': [segmenter._masks[0]],
            'sky': [],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Should use top edge of planet (row 60)
        assert len(limb) == 100
        assert np.allclose(limb, 60.0)
    
    def test_combine_masks_varying_horizon(self):
        """Test limb extraction with non-horizontal horizon."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def slanted_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            planet_mask = np.zeros((h, w), dtype=bool)
            
            # Create slanted horizon: starts at row 20, ends at row 80
            for col in range(w):
                horizon_row = int(20 + 60 * col / w)
                sky_mask[:horizon_row, col] = True
                planet_mask[horizon_row:, col] = True
            
            return [
                {'mask': sky_mask, 'area': np.sum(sky_mask)},
                {'mask': planet_mask, 'area': np.sum(planet_mask)}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=slanted_segmenter, interactive=False
        )
        segmenter._masks = slanted_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Check limb follows the slant
        assert limb[0] < 25  # Start low
        assert limb[-1] > 75  # End high
        assert limb[-1] > limb[0]  # Increasing
    
    def test_combine_masks_with_outliers(self):
        """Test outlier detection and interpolation in limb extraction."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def outlier_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            planet_mask = np.zeros((h, w), dtype=bool)
            
            # Normal horizon at row 50, except one outlier column
            for col in range(w):
                if col == 50:  # Single outlier
                    horizon_row = 10  # Way off
                else:
                    horizon_row = 50  # Normal
                
                sky_mask[:horizon_row, col] = True
                planet_mask[horizon_row:, col] = True
            
            return [
                {'mask': sky_mask, 'area': np.sum(sky_mask)},
                {'mask': planet_mask, 'area': np.sum(planet_mask)}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=outlier_segmenter, interactive=False
        )
        segmenter._masks = outlier_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Single outlier should be detected and interpolated
        # Most of limb should be around 49.5 (normal horizon)
        assert np.median(limb) > 45  # Most values near 49.5
        # The outlier column should be interpolated to be close to neighbors
        assert 45 < limb[50] < 55  # Interpolated, not 9.5


class TestMaskSegmenterPipeline:
    """Test complete segmentation pipeline with downsampling."""
    
    def test_segment_with_downsampling(self):
        """Test complete pipeline with downsampling."""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        
        def test_segmenter(img):
            h, w = img.shape[:2]
            # Create simple upper/lower masks
            sky = np.zeros((h, w), dtype=bool)
            sky[:h//2, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[h//2:, :] = True
            return [
                {'mask': sky, 'area': np.sum(sky)},
                {'mask': planet, 'area': np.sum(planet)}
            ]
        
        segmenter = MaskSegmenter(
            image,
            method="custom",
            segment_fn=test_segmenter,
            downsample_factor=2,
            interactive=False
        )
        
        limb = segmenter.segment()
        
        # Should return limb for original size (200 wide)
        assert len(limb) == 200
        # Limb should be around row 100 (midpoint)
        # Actual value will be 99.5 due to indexing
        assert 95 < np.mean(limb) < 105
    
    def test_segment_no_downsampling(self):
        """Test pipeline without downsampling."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def test_segmenter(img):
            h, w = img.shape[:2]
            sky = np.zeros((h, w), dtype=bool)
            sky[:40, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[60:, :] = True
            return [
                {'mask': sky, 'area': np.sum(sky)},
                {'mask': planet, 'area': np.sum(planet)}
            ]
        
        segmenter = MaskSegmenter(
            image,
            method="custom",
            segment_fn=test_segmenter,
            downsample_factor=1,
            interactive=False
        )
        
        limb = segmenter.segment()
        
        assert len(limb) == 100
        # Sky ends at 39, planet starts at 60
        # Limb = (39 + 60) / 2 = 49.5
        assert np.allclose(limb, 49.5)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_combine_masks_missing_column(self):
        """Test limb extraction when some columns have no masks."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def sparse_segmenter(img):
            h, w = img.shape[:2]
            sky_mask = np.zeros((h, w), dtype=bool)
            planet_mask = np.zeros((h, w), dtype=bool)
            
            # Only fill even columns
            for col in range(0, w, 2):
                sky_mask[:40, col] = True
                planet_mask[60:, col] = True
            
            return [
                {'mask': sky_mask, 'area': np.sum(sky_mask)},
                {'mask': planet_mask, 'area': np.sum(planet_mask)}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=sparse_segmenter, interactive=False
        )
        segmenter._masks = sparse_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        limb = result['limb']
        
        # Should interpolate missing columns
        assert len(limb) == 100
        assert not np.any(np.isnan(limb))  # No NaNs left
        assert np.allclose(limb, 49.5, atol=5)  # Close to expected
    
    def test_combine_masks_returns_all_components(self):
        """Test that _combine_masks returns limb, planet_mask, and sky_mask."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def simple_segmenter(img):
            h, w = img.shape[:2]
            sky = np.zeros((h, w), dtype=bool)
            sky[:50, :] = True
            planet = np.zeros((h, w), dtype=bool)
            planet[50:, :] = True
            return [
                {'mask': sky, 'area': 5000},
                {'mask': planet, 'area': 5000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=simple_segmenter, interactive=False
        )
        segmenter._masks = simple_segmenter(image)
        
        classified = {
            'sky': [segmenter._masks[0]],
            'planet': [segmenter._masks[1]],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        
        assert 'limb' in result
        assert 'planet_mask' in result
        assert 'sky_mask' in result
        assert isinstance(result['limb'], np.ndarray)
        assert isinstance(result['planet_mask'], np.ndarray)
        assert isinstance(result['sky_mask'], np.ndarray)
    
    def test_fallback_when_no_classifications(self):
        """Test fallback behavior when masks aren't classified."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        def two_mask_segmenter(img):
            h, w = img.shape[:2]
            mask1 = np.zeros((h, w), dtype=bool)
            mask1[:50, :] = True
            mask2 = np.zeros((h, w), dtype=bool)
            mask2[50:, :] = True
            return [
                {'mask': mask1, 'area': 5000},
                {'mask': mask2, 'area': 5000}
            ]
        
        segmenter = MaskSegmenter(
            image, method="custom", segment_fn=two_mask_segmenter, interactive=False
        )
        segmenter._masks = two_mask_segmenter(image)
        
        # Pass empty classifications - should use fallback
        classified = {
            'sky': [],
            'planet': [],
            'exclude': []
        }
        
        result = segmenter._combine_masks(classified)
        
        # Should still produce a limb using masks[0] and masks[1]
        assert 'limb' in result
        assert len(result['limb']) == 100