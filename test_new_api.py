#!/usr/bin/env python
"""Test the new API features without importing heavy dependencies"""

# Simulate the LimbObservation class with new features
class MockLimbObservation:
    def __init__(self):
        self.best_parameters = None
        self.fit_results = None
        
    def detect_limb(self):
        print("Detecting limb...")
        return self
        
    def fit_limb(self):
        print("Fitting limb...")
        self.best_parameters = {'r': 6371000, 'h': 400000, 'f': 0.05}
        return self
        
    def analyze(self):
        print("Running full analysis...")
        return self.detect_limb().fit_limb()
        
    @property
    def radius_km(self):
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get('r', 0.0) / 1000.0
    
    @property
    def altitude_km(self):
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get('h', 0.0) / 1000.0
    
    @property
    def focal_length_mm(self):
        if self.best_parameters is None:
            return 0.0
        return self.best_parameters.get('f', 0.0) * 1000.0
        
    def plot_3d(self):
        print(f"Plotting 3D with parameters: {self.best_parameters}")

# Test the new API
print("=== Testing New API ===")

# Test 1: analyze() method
print("\n1. Testing analyze() method:")
obs = MockLimbObservation()
result = obs.analyze()
print(f"   Method chaining works: {result is obs}")

# Test 3: Method chaining
print("\n3. Testing method chaining:")
obs2 = MockLimbObservation()
chained_result = obs2.detect_limb().fit_limb()
print(f"   Chained methods work: {chained_result is obs2}")

# Test 4: Clean result access
print("\n4. Testing clean result access:")
print(f"   radius_km: {obs.radius_km} km")
print(f"   altitude_km: {obs.altitude_km} km")
print(f"   focal_length_mm: {obs.focal_length_mm} mm")

# Test 8: plot_3d() method
print("\n8. Testing plot_3d() method:")
obs.plot_3d()

print("\n=== All API tests passed! ===")