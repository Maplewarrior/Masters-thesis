import pytest
import torch
import torch.nn as nn
import pdb
import numpy as np
from src.evaluation.unlearning_evaluator import UnlearningEvaluator
import torch.distributions as td

class MockModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = logits
    
    def inference(self, x):
        return {'logits': self.logits}

class MockDataLoader:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __iter__(self):
        yield self.x, self.y

@pytest.fixture
def evaluator():
    return UnlearningEvaluator()

def create_test_data(batch_size=4, num_classes=3):
    # Create one-hot encoded labels
    y = torch.zeros(batch_size, num_classes)
    y[range(batch_size), range(batch_size % num_classes)] = 1
    
    # Create mock input (not used in tests but needed for dataloader)
    x = torch.randn(batch_size, 10)
    
    return x, y

def create_identical_predictions(batch_size=4, num_classes=3):
    logits = torch.tensor([
        [10.0, -10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, 10.0],
        [10.0, -10.0, -10.0],
    ])
    return logits, logits

def create_completely_different_predictions(batch_size=4, num_classes=3):
    logits_1 = torch.tensor([
        [-10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0],
    ])
    logits_2 = torch.tensor([
        [-10.0, -10.0, 10.0],
        [-10.0, -10.0, 10.0],
        [-10.0, -10.0, 10.0],
        [-10.0, -10.0, 10.0],
    ])
    return logits_1, logits_2


@pytest.mark.parametrize("metric,bounds", [
    ("Hamming PD", (0, 1)),
    ("min max normalized HPD", (0, 1)),
    ("avg norm prediction difference", (0, 1)),
    ("KL divergence", (0, float('inf'))),
    ("JS divergence", (0, 1.0)),
])
def test_metric_bounds(evaluator, metric, bounds):
    x, y = create_test_data()
    
    # Test lower bound with identical predictions
    logits_1, logits_2 = create_identical_predictions()
    model_1 = MockModel(logits_1)
    model_2 = MockModel(logits_2)
    dataloader = MockDataLoader(x, y)
    
    result = evaluator.evaluate(model_1, model_2, [metric], dataloader)
    assert np.isclose(result[metric], bounds[0], atol=1e-4), f"{metric} not close to lower bound for edge case input. Expected {bounds[0]} got {result[metric]}"
    
    # Test upper bound with completely different predictions
    logits_1, logits_2 = create_completely_different_predictions()
    model_1 = MockModel(logits_1 * 100)
    model_2 = MockModel(logits_2 * 100)
    dataloader = MockDataLoader(x, y)
    
    result = evaluator.evaluate(model_1, model_2, [metric], dataloader)

    if bounds[1] != torch.inf:
        assert np.isclose(result[metric], bounds[1], atol=1e-4), f"{metric} not close to upper bound for edge case input. Expected {bounds[1]} got {result[metric]}"

def test_accuracy():
    evaluator = UnlearningEvaluator()
    
    # Create perfect predictions
    y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    accuracy = evaluator.calculate_accuracy(preds=y * 10, y=y)
    assert accuracy == 1.0, "Perfect predictions should give 100% accuracy"
    
    # Create completely wrong predictions
    wrong_preds = torch.tensor([[-100.0, -100.0, 100.0],
                              [100.0, -100.0, -100.0],
                              [-100.0, 100.0, -100.0]])
    
    accuracy = evaluator.calculate_accuracy(wrong_preds, y)
    assert accuracy == 0.0, "Completely wrong predictions should give % accuracy"


@pytest.mark.parametrize("metric", [
    "Hamming PD",
    "min max normalized HPD",
    "avg norm prediction difference",
    "JS divergence",
])
def test_metric_symmetry(evaluator, metric):
    x, y = create_test_data()
    
    # Create two different sets of predictions
    logits_1 = torch.tensor([
        [5.0, -2.0, -3.0],
        [-1.0, 4.0, -3.0],
        [-2.0, -1.0, 3.0],
        [4.0, -2.0, -2.0],
    ])
    logits_2 = torch.tensor([
        [-2.0, 4.0, -2.0],
        [3.0, -2.0, -1.0],
        [-3.0, 4.0, -1.0],
        [-2.0, -1.0, 3.0],
    ])
    
    model_1 = MockModel(logits_1)
    model_2 = MockModel(logits_2)
    dataloader = MockDataLoader(x, y)
    
    # Test both directions
    result_1 = evaluator.evaluate(model_1, model_2, [metric], dataloader)
    result_2 = evaluator.evaluate(model_2, model_1, [metric], dataloader)
    
    assert abs(result_1[metric] - result_2[metric]) < 1e-5, f"{metric} should be symmetric"

def test_batch_independence():
    """Test that metrics give consistent results regardless of batch size"""
    evaluator = UnlearningEvaluator()
    
    # Create predictions for 4 samples
    preds_1 = torch.tensor([
        [5.0, -2.0], 
        [-1.0, 4.0],
        [3.0, -3.0],
        [-2.0, 5.0]
    ])
    preds_2 = torch.tensor([
        [-2.0, 4.0],
        [3.0, -2.0],
        [-3.0, 4.0],
        [4.0, -1.0]
    ])
    
    # Calculate metrics with different batch sizes
    single_batch_js = evaluator.JS_divergence(preds_1, preds_2)
    # Split into two batches
    batch1_js = evaluator.JS_divergence(preds_1[:2], preds_2[:2])
    batch2_js = evaluator.JS_divergence(preds_1[2:], preds_2[2:])
    avg_js = (batch1_js + batch2_js) / 2
    
    assert abs(single_batch_js - avg_js) < 1e-5, "JS divergence should be consistent across different batch sizes"

def test_numerical_stability():
    """Test metrics with extreme values"""
    evaluator = UnlearningEvaluator()
    
    # Test with very large logits
    large_preds_1 = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]])
    large_preds_2 = torch.tensor([[-1000.0, 1000.0], [1000.0, -1000.0]])
    
    kl_div = evaluator.KL_divergence(large_preds_1, large_preds_2)
    js_div = evaluator.JS_divergence(large_preds_1, large_preds_2)
    
    assert not torch.isnan(torch.tensor(kl_div)), "KL divergence should handle large values"
    assert not torch.isnan(torch.tensor(js_div)), "JS divergence should handle large values"
    
    # Test with very small differences
    small_diff_1 = torch.tensor([[0.1 + 1e-7, 0.0], [0.0, 0.1 - 1e-7]])
    small_diff_2 = torch.tensor([[0.1, 0.0], [0.0, 0.1]])
    
    kl_div = evaluator.KL_divergence(small_diff_1, small_diff_2)
    js_div = evaluator.JS_divergence(small_diff_1, small_diff_2)
    
    assert kl_div >= 0, "KL divergence should handle small differences"
    assert js_div >= 0, "JS divergence should handle small differences"

@pytest.mark.parametrize("metric", [
    "Hamming PD",
    "Error corrected HPD",
    "min max normalized HPD",
    # "avg norm prediction difference",
    "KL divergence",
    "JS divergence"
])
def test_metric_scale_invariance(evaluator, metric):
    """Test that metrics are invariant to scaling of logits"""
    x, y = create_test_data()
    
    # Create base predictions
    logits_1 = torch.tensor([
        [5.0, -2.0, -3.0],
        [-1.0, 4.0, -3.0],
        [-2.0, -1.0, 3.0],
        [4.0, -2.0, -2.0],
    ])
    logits_2 = torch.tensor([
        [-2.0, 4.0, -2.0],
        [3.0, -2.0, -1.0],
        [-3.0, 4.0, -1.0],
        [-2.0, -1.0, 3.0],
    ])

    # Create scaled versions
    scale = 10.0
    scaled_logits_1 = logits_1 * scale
    scaled_logits_2 = logits_2 * scale

    mvn1 = td.MultivariateNormal(loc=torch.tensor([1., 2., 3.]), covariance_matrix=torch.eye(3) * 2)
    mvn2 = td.MultivariateNormal(loc=torch.tensor([-0.5, 2.7, 5.]), covariance_matrix=torch.eye(3) * 0.5)
    
    m, b = 4, 2
    transform = td.AffineTransform(m, b)
    scaled_mvn1 = td.TransformedDistribution(mvn1, [transform])
    scaled_mvn2 = td.TransformedDistribution(mvn2, [transform])

    if metric not in ['KL divergence', 'JS divergence']:
        model_1 = MockModel(logits_1)
        model_2 = MockModel(logits_2)
        scaled_model_1 = MockModel(scaled_logits_1)
        scaled_model_2 = MockModel(scaled_logits_2)

    else:
        n = 10000 # number of samples

        x_samples = mvn1.sample(sample_shape=(n,))

        logits_p = mvn1.log_prob(x_samples)
        logits_q = mvn2.log_prob(x_samples)
        
        x_scaled_samples = scaled_mvn1.sample((n,))
        logits_p_scaled = scaled_mvn1.log_prob(x_scaled_samples)
        logits_q_scaled = scaled_mvn2.log_prob(x_scaled_samples)

        model_1 = MockModel(logits_p)
        model_2 = MockModel(logits_q)
        scaled_model_1 = MockModel(logits_p_scaled)
        scaled_model_2 = MockModel(logits_q_scaled) 
    
    dataloader = MockDataLoader(x, y)
    # Compare results
    result_1 = evaluator.evaluate(model_1, model_2, [metric], dataloader)
    result_2 = evaluator.evaluate(scaled_model_1, scaled_model_2, [metric], dataloader)

    assert abs(result_1[metric] - result_2[metric]) < 1e-4, f"{metric} should be invariant to scaling"

def test_error_corrected_HPD_edge_cases():
    """Test error-corrected Hamming Prediction Distance with edge cases"""
    evaluator = UnlearningEvaluator()
    
    # Create data where unlearned model is perfect
    x = torch.randn(4, 3)
    y = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    perfect_preds = torch.tensor([[100.0, -100.0, -100.0],
                                [-100.0, 100.0, -100.0],
                                [-100.0, -100.0, 100.0],
                                [100.0, -100.0, -100.0]])
    different_preds = torch.tensor([[-100.0, 100.0, -100.0],
                                  [100.0, -100.0, -100.0],
                                  [100.0, -100.0, -100.0],
                                  [-100.0, -100.0, 100.0]])
    
    model_1 = MockModel(perfect_preds)
    model_2 = MockModel(different_preds)
    dataloader = MockDataLoader(x, y)
    
    # with pytest.raises(Exception):
        # Should handle division by zero when unlearned model has perfect accuracy
    result = evaluator.evaluate(model_1, model_2, ["Error corrected HPD"], dataloader)
    
    assert result['Error corrected HPD'] == torch.inf, 'Division by zero due to a perfect model does not return torch.inf'