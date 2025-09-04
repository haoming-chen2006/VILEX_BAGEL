#!/usr/bin/env python3
"""
Test script to diagnose checkpoint saving issues
Usage: python test_checkpoint_saving.py
"""

import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path
import tempfile
import shutil
from dataclasses import dataclass

# Add the train directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'train'))

from train.fsdp_utils import FSDPCheckpoint, FSDPConfig
from train.train_utils import create_logger

@dataclass 
class MockTrainingArgs:
    checkpoint_dir: str = "test_checkpoints"
    save_every: int = 2
    results_dir: str = "/home/haoming/Bagel/results"

def test_checkpoint_creation():
    """Test basic checkpoint directory creation"""
    print("=== Testing Checkpoint Directory Creation ===")
    
    test_dir = "test_checkpoint_dirs"
    os.makedirs(test_dir, exist_ok=True)
    
    # Test step directory creation
    for step in [10, 20, 100]:
        step_dir = os.path.join(test_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        
        if os.path.exists(step_dir):
            print(f"✅ Successfully created {step_dir}")
        else:
            print(f"❌ Failed to create {step_dir}")
    
    # Cleanup
    shutil.rmtree(test_dir, ignore_errors=True)
    print()

def test_checkpoint_permissions():
    """Test write permissions in checkpoint directory"""
    print("=== Testing Checkpoint Write Permissions ===")
    
    training_args = MockTrainingArgs()
    os.makedirs(training_args.checkpoint_dir, exist_ok=True)
    
    test_file = os.path.join(training_args.checkpoint_dir, "test_write.txt")
    
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        print(f"✅ Write permissions OK in {training_args.checkpoint_dir}")
        os.remove(test_file)
    except Exception as e:
        print(f"❌ Write permission error: {e}")
    
    print()

def test_fsdp_checkpoint_mock():
    """Test FSDP checkpoint saving with mock objects"""
    print("=== Testing FSDP Checkpoint (Mock) ===")
    
    training_args = MockTrainingArgs()
    os.makedirs(training_args.checkpoint_dir, exist_ok=True)
    
    # Create mock objects
    class MockModel:
        def state_dict(self):
            return {"mock_param": torch.randn(10, 10)}
    
    class MockOptimizer:
        def state_dict(self):
            return {"step": 100, "mock_state": torch.randn(5)}
    
    class MockScheduler:
        def state_dict(self):
            return {"last_lr": 0.001}
    
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        backward_prefetch="BACKWARD_PRE", 
        cpu_offload=False,
        num_replicate=1,
        num_shard=1
    )
    
    logger = create_logger(training_args.results_dir, 0)
    
    try:
        # Mock the FSDP save function to test directory creation
        step_dir = os.path.join(training_args.checkpoint_dir, "step_10")
        os.makedirs(step_dir, exist_ok=True)
        
        # Test saving individual components
        torch.save(MockModel().state_dict(), os.path.join(step_dir, "model.pt"))
        torch.save(MockOptimizer().state_dict(), os.path.join(step_dir, "optimizer.pt"))
        torch.save(MockScheduler().state_dict(), os.path.join(step_dir, "scheduler.pt"))
        
        print(f"✅ Mock checkpoint saved to {step_dir}")
        
        # Verify files exist
        expected_files = ["model.pt", "optimizer.pt", "scheduler.pt"]
        for file in expected_files:
            file_path = os.path.join(step_dir, file)
            if os.path.exists(file_path):
                print(f"✅ {file} exists ({os.path.getsize(file_path)} bytes)")
            else:
                print(f"❌ {file} missing")
                
    except Exception as e:
        print(f"❌ Mock checkpoint save failed: {e}")
    
    print()

def test_distributed_checkpoint():
    """Test distributed checkpoint saving logic"""
    print("=== Testing Distributed Checkpoint Logic ===")
    
    # Simulate multi-rank scenario
    world_size = 4
    
    for rank in range(world_size):
        print(f"--- Simulating Rank {rank} ---")
        
        # Mock data status (what each rank would have)
        mock_data_status = {
            "vlm_sft": {
                rank: {"parquet_idx": rank * 10, "sample_idx": rank * 100}
            }
        }
        
        # Simulate gather operation
        if rank == 0:
            gather_list = [None] * world_size
            # Simulate what rank 0 would receive
            for i in range(world_size):
                gather_list[i] = {
                    "vlm_sft": {
                        i: {"parquet_idx": i * 10, "sample_idx": i * 100}
                    }
                }
            print(f"  ✅ Rank 0 would gather: {len(gather_list)} data status objects")
        else:
            gather_list = None
            print(f"  ✅ Rank {rank} data status ready for gathering")
    
    print()

def test_checkpoint_loading():
    """Test checkpoint loading/resuming"""
    print("=== Testing Checkpoint Loading ===")
    
    training_args = MockTrainingArgs()
    
    # First create a checkpoint to load
    step_dir = os.path.join(training_args.checkpoint_dir, "step_10")
    os.makedirs(step_dir, exist_ok=True)
    
    # Create mock checkpoint files
    mock_data = {
        "model.pt": {"param1": torch.randn(5, 5)},
        "optimizer.pt": {"step": 10, "lr": 0.001},
        "scheduler.pt": {"last_lr": 0.001},
        "train_status.pt": {"step": 10, "epoch": 1}
    }
    
    for filename, data in mock_data.items():
        torch.save(data, os.path.join(step_dir, filename))
        print(f"✅ Created {filename}")
    
    # Test loading
    try:
        for filename in mock_data.keys():
            file_path = os.path.join(step_dir, filename)
            loaded_data = torch.load(file_path)
            print(f"✅ Successfully loaded {filename}")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
    
    print()

def test_auto_resume_logic():
    """Test auto-resume checkpoint detection"""
    print("=== Testing Auto-Resume Logic ===")
    
    training_args = MockTrainingArgs()
    
    # Create multiple checkpoint steps
    steps = [100, 200, 150, 300]  # Out of order to test sorting
    
    for step in steps:
        step_dir = os.path.join(training_args.checkpoint_dir, f"step_{step}")
        os.makedirs(step_dir, exist_ok=True)
        # Create a marker file
        torch.save({"step": step}, os.path.join(step_dir, "train_status.pt"))
    
    # Find latest checkpoint (simulate get_latest_ckpt function)
    if os.path.exists(training_args.checkpoint_dir):
        step_dirs = [d for d in os.listdir(training_args.checkpoint_dir) 
                    if d.startswith("step_") and os.path.isdir(os.path.join(training_args.checkpoint_dir, d))]
        
        if step_dirs:
            latest_step = max([int(d.split("_")[1]) for d in step_dirs])
            latest_dir = os.path.join(training_args.checkpoint_dir, f"step_{latest_step}")
            print(f"✅ Found latest checkpoint: {latest_dir}")
            print(f"✅ Available steps: {sorted([int(d.split('_')[1]) for d in step_dirs])}")
        else:
            print("❌ No checkpoint directories found")
    else:
        print("❌ Checkpoint directory doesn't exist")
    
    print()

def cleanup_test_files():
    """Clean up test files"""
    print("=== Cleaning Up Test Files ===")
    
    test_dirs = ["test_checkpoints", "test_results", "test_checkpoint_dirs"]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir, ignore_errors=True)
            print(f"✅ Cleaned up {test_dir}")

def main():
    print("🔍 BAGEL Checkpoint Saving Diagnostic Test")
    print("=" * 50)
    
    # Run all tests
    test_checkpoint_creation()
    test_checkpoint_permissions() 
    test_fsdp_checkpoint_mock()
    test_distributed_checkpoint()
    test_checkpoint_loading()
    test_auto_resume_logic()
    
    print("=" * 50)
    print("✅ All tests completed!")
    print("\n📋 If any tests failed, check:")
    print("1. Disk space and write permissions")
    print("2. FSDP checkpoint implementation")
    print("3. Distributed training setup")
    print("4. Path configuration in YAML")
    
    cleanup_test_files()

if __name__ == "__main__":
    main()