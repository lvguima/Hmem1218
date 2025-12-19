
import torch

def test_list_remove_dict_tensor():
    print("Testing list.remove with dict containing tensors...")
    
    # Create a list of dicts with tensors
    pending_updates = []
    
    update1 = {
        'step': 1,
        'pogt': torch.randn(1, 10),
        'prediction': torch.randn(1, 10)
    }
    
    update2 = {
        'step': 2,
        'pogt': torch.randn(1, 10),
        'prediction': torch.randn(1, 10)
    }
    
    pending_updates.append(update1)
    pending_updates.append(update2)
    
    print(f"Initial length: {len(pending_updates)}")
    
    # Simulate the logic in exp_hmem.py
    ready_updates = [u for u in pending_updates if u['step'] == 1]
    
    print(f"Ready updates: {len(ready_updates)}")
    
    try:
        for update in ready_updates:
            print("Attempting to remove update...")
            pending_updates.remove(update)
            print("Remove successful.")
    except Exception as e:
        print(f"\nCRITICAL FAILURE: {type(e).__name__}: {e}")
        return False

    print(f"Final length: {len(pending_updates)}")
    return True

if __name__ == "__main__":
    if test_list_remove_dict_tensor():
        print("\nTest PASSED: list.remove works with tensor dicts (unexpectedly!).")
    else:
        print("\nTest FAILED: list.remove crashes as expected.")
