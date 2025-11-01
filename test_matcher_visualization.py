import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import json
from libs_week4.descriptor import ORBDescriptor, DaisyDescriptor, SIFTDescriptor, DescriptorMatcher

def load_image_from_bbdd(image_id: int, bbdd_folder: str = "BBDD") -> tuple[np.ndarray, str]:
    """Load an image from BBDD folder."""
    image_path = Path(bbdd_folder) / f"bbdd_{image_id:05d}.jpg"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    return image_norm, str(image_path)

def load_query_image(query_id: int, query_folder: str = "qsd1_w4") -> tuple[np.ndarray, str]:
    """
    Load a query image.
    
    Args:
        query_id: ID of the query image (e.g., 0 for 00000.jpg)
        query_folder: Path to query folder
        
    Returns:
        Tuple of (image, image_path)
    """
    image_path = Path(query_folder) / f"{query_id:05d}.jpg"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to load query image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    
    return image_norm, str(image_path)

def get_descriptor_cache_path(bbdd_folder: str, descriptor_type: str) -> Path:
    """
    Get the cache file path for pre-computed descriptors.
    
    Args:
        bbdd_folder: Path to BBDD folder
        descriptor_type: 'orb', 'daisy', or 'sift'
        
    Returns:
        Path to cache file
    """
    cache_dir = Path(bbdd_folder) / "descriptor_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / f"{descriptor_type}_descriptors.pkl"

def precompute_bbdd_descriptors(bbdd_folder: str = "BBDD",
                                descriptor_type: str = "orb",
                                force_recompute: bool = False) -> dict:
    """
    Pre-compute and cache descriptors for all BBDD images.
    
    Args:
        bbdd_folder: Path to BBDD folder
        descriptor_type: 'orb', 'daisy', or 'sift'
        force_recompute: If True, recompute even if cache exists
        
    Returns:
        Dictionary mapping image_id -> (keypoints, descriptors)
    """
    cache_path = get_descriptor_cache_path(bbdd_folder, descriptor_type)
    
    # Load from cache if exists and not forcing recompute
    if cache_path.exists() and not force_recompute:
        print(f"Loading pre-computed {descriptor_type.upper()} descriptors from cache...")
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        print(f"  ✓ Loaded {len(cached_data)} pre-computed descriptors from cache")
        return cached_data
    
    # Initialize descriptor
    print(f"Pre-computing {descriptor_type.upper()} descriptors for BBDD...")
    if descriptor_type.lower() == 'orb':
        descriptor = ORBDescriptor(n_features=500, scale_factor=1.2, n_levels=8)
    elif descriptor_type.lower() == 'daisy':
        descriptor = DaisyDescriptor(step=16, radius=15, rings=3, histograms=8, orientations=8)
    elif descriptor_type.lower() == 'sift':
        descriptor = SIFTDescriptor(n_features=0)
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")
    
    # Get all BBDD images
    bbdd_path = Path(bbdd_folder)
    image_files = sorted(bbdd_path.glob("bbdd_*.jpg"))
    num_images = len(image_files)
    
    print(f"  Processing {num_images} images...")
    
    # Pre-compute descriptors
    descriptor_cache = {}
    
    for idx, image_file in enumerate(image_files):
        # Extract image ID from filename
        bbdd_id = int(image_file.stem.split('_')[1])
        
        try:
            # Load image
            image, _ = load_image_from_bbdd(bbdd_id, bbdd_folder)
            
            # Compute descriptors
            kp, desc = descriptor.detect_and_compute(image)
            
            # Convert keypoints to serializable format
            # (cv2.KeyPoint objects can't be pickled directly)
            kp_data = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
            
            descriptor_cache[bbdd_id] = {
                'keypoints_data': kp_data,
                'descriptors': desc
            }
            
            if (idx + 1) % 50 == 0:
                print(f"    Processed {idx + 1}/{num_images} images...")
                
        except Exception as e:
            print(f"  Warning: Failed to process image {bbdd_id}: {e}")
            continue
    
    # Save to cache
    print(f"\n  Saving {len(descriptor_cache)} descriptors to cache...")
    with open(cache_path, 'wb') as f:
        pickle.dump(descriptor_cache, f)
    
    # Save metadata
    metadata_path = cache_path.with_suffix('.json')
    metadata = {
        'descriptor_type': descriptor_type,
        'num_images': len(descriptor_cache),
        'descriptor_params': descriptor.to_dict() if hasattr(descriptor, 'to_dict') else {}
    }
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Cache saved to: {cache_path}")
    print(f"  ✓ Metadata saved to: {metadata_path}")
    
    return descriptor_cache

def reconstruct_keypoints(kp_data: list) -> list:
    """
    Reconstruct cv2.KeyPoint objects from serialized data.
    
    Args:
        kp_data: List of tuples (pt, size, angle, response, octave, class_id)
        
    Returns:
        List of cv2.KeyPoint objects
    """
    keypoints = []
    for pt, size, angle, response, octave, class_id in kp_data:
        kp = cv2.KeyPoint(
            x=pt[0], y=pt[1],
            size=size,
            angle=angle,
            response=response,
            octave=octave,
            class_id=class_id
        )
        keypoints.append(kp)
    return keypoints

def match_query_to_bbdd(query_image: np.ndarray, 
                        bbdd_folder: str = "BBDD",
                        descriptor_type: str = "orb",
                        matcher_type: str = "BF",
                        top_k: int = 5,
                        use_cache: bool = True) -> list[tuple[int, int, float]]:
    """
    Match a query image against all images in BBDD using pre-computed descriptors.
    
    Args:
        query_image: Query image (RGB, [0,1])
        bbdd_folder: Path to BBDD folder
        descriptor_type: 'orb', 'daisy', or 'sift'
        matcher_type: 'BF' or 'FLANN'
        top_k: Number of top matches to return
        use_cache: If True, use pre-computed descriptors from cache
        
    Returns:
        List of tuples (bbdd_id, num_matches, match_score)
    """
    # Initialize descriptor
    if descriptor_type.lower() == 'orb':
        descriptor = ORBDescriptor(n_features=500, scale_factor=1.2, n_levels=8)
        norm_type = cv2.NORM_HAMMING
    elif descriptor_type.lower() == 'daisy':
        descriptor = DaisyDescriptor(step=16, radius=15, rings=3, histograms=8, orientations=8)
        norm_type = cv2.NORM_L2
    elif descriptor_type.lower() == 'sift':
        descriptor = SIFTDescriptor(n_features=0)
        norm_type = cv2.NORM_L2
    else:
        raise ValueError(f"Unknown descriptor type: {descriptor_type}")
    
    # Initialize matcher
    matcher = DescriptorMatcher(
        matcher_type=matcher_type,
        norm_type=norm_type,
        cross_check=False,
        ratio_test_threshold=0.75
    )
    
    # Compute query descriptors
    print(f"Computing {descriptor_type.upper()} descriptors for query image...")
    query_kp, query_desc = descriptor.detect_and_compute(query_image)
    print(f"  → Query: {len(query_kp)} keypoints, descriptor shape: {query_desc.shape if query_desc is not None else 'None'}")
    
    if query_desc is None or len(query_desc) == 0:
        print("ERROR: No descriptors found in query image!")
        return []
    
    # Load or compute BBDD descriptors
    if use_cache:
        descriptor_cache = precompute_bbdd_descriptors(bbdd_folder, descriptor_type, force_recompute=False)
        print(f"\nMatching against {len(descriptor_cache)} cached BBDD descriptors...")
    else:
        print("\nComputing BBDD descriptors on-the-fly (cache disabled)...")
        bbdd_path = Path(bbdd_folder)
        num_bbdd_images = len(list(bbdd_path.glob("bbdd_*.jpg")))
        descriptor_cache = {}
        for bbdd_id in range(num_bbdd_images):
            bbdd_image, _ = load_image_from_bbdd(bbdd_id, bbdd_folder)
            kp, desc = descriptor.detect_and_compute(bbdd_image)
            kp_data = [(k.pt, k.size, k.angle, k.response, k.octave, k.class_id) for k in kp]
            descriptor_cache[bbdd_id] = {'keypoints_data': kp_data, 'descriptors': desc}
    
    # Match against all BBDD images
    match_results = []
    
    for bbdd_id, cached_data in descriptor_cache.items():
        try:
            bbdd_desc = cached_data['descriptors']
            
            if bbdd_desc is None or len(bbdd_desc) == 0:
                continue
            
            # Match descriptors
            matches = matcher.match(query_desc, bbdd_desc)
            num_matches = len(matches)
            
            # Calculate match score (average distance of good matches)
            if num_matches > 0:
                avg_distance = np.mean([m.distance for m in matches])
                match_score = num_matches / (1.0 + avg_distance)  # Higher is better
            else:
                match_score = 0.0
            
            match_results.append((bbdd_id, num_matches, match_score))
            
            if (bbdd_id + 1) % 50 == 0:
                print(f"  Processed {bbdd_id + 1}/{len(descriptor_cache)} images...")
                
        except Exception as e:
            print(f"  Warning: Failed to process BBDD image {bbdd_id}: {e}")
            continue
    
    # Sort by number of matches (descending)
    match_results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMatching complete! Found matches for {len(match_results)} images.")
    
    return match_results[:top_k]

def visualize_matches(query_image: np.ndarray,
                     query_kp: list,
                     bbdd_image: np.ndarray,
                     bbdd_kp: list,
                     matches: list,
                     query_id: int,
                     bbdd_id: int):
    """
    Visualize keypoint matches between query and BBDD image.
    
    Args:
        query_image: Query image (RGB, [0,1])
        query_kp: Query keypoints
        bbdd_image: BBDD image (RGB, [0,1])
        bbdd_kp: BBDD keypoints
        matches: List of cv2.DMatch objects
        query_id: Query image ID
        bbdd_id: BBDD image ID
    """
    # Convert to uint8 for OpenCV
    query_uint8 = (query_image * 255).astype(np.uint8)
    bbdd_uint8 = (bbdd_image * 255).astype(np.uint8)
    
    # Draw matches
    match_img = cv2.drawMatches(
        query_uint8, query_kp,
        bbdd_uint8, bbdd_kp,
        matches[:50],  # Show top 50 matches for clarity
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    )
    
    # Display
    plt.figure(figsize=(20, 10))
    plt.imshow(match_img)
    
    title = f"Query {query_id:05d} → BBDD {bbdd_id:05d} | {len(matches)} matches"
    plt.title(title, fontsize=16, fontweight='bold')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def classify_match_confidence(match_results: list[tuple[int, int, float]], 
                               min_matches: int = 30,
                               gap_ratio: float = 2.0,
                               min_score: float = 5.0) -> tuple[str, float]:
    """
    Classify match confidence level.
    
    Args:
        match_results: List of (bbdd_id, num_matches, score)
        min_matches: Minimum absolute matches
        gap_ratio: Minimum ratio between top 2 matches
        min_score: Minimum match score
        
    Returns:
        Tuple of (confidence_level, confidence_score)
        - confidence_level: 'HIGH', 'MEDIUM', 'LOW', 'NO_MATCH'
        - confidence_score: 0.0 to 1.0
    """
    if not match_results or match_results[0][1] == 0:
        return 'NO_MATCH', 0.0
    
    top_match = match_results[0]
    bbdd_id, num_matches, score = top_match
    
    # Check 1: Absolute match count
    if num_matches < min_matches:
        return 'NO_MATCH', 0.0
    
    # Check 2: Match score quality
    if score < min_score:
        return 'LOW', 0.3
    
    # Check 3: Gap between top 2 matches
    if len(match_results) >= 2:
        second_matches = match_results[1][1]
        if second_matches > 0:
            ratio = num_matches / second_matches
            
            if ratio < gap_ratio:
                # Top match not significantly better
                return 'LOW', 0.4
            elif ratio < gap_ratio * 1.5:
                return 'MEDIUM', 0.6
            else:
                # Clear winner
                return 'HIGH', 0.9
    
    # Only one match or very few matches
    if num_matches < min_matches * 2:
        return 'MEDIUM', 0.6
    
    return 'HIGH', 0.9


def test_query_matching(query_id: int = 0,
                       query_folder: str = "qsd1_w4",
                       bbdd_folder: str = "BBDD",
                       descriptor_type: str = "orb",
                       top_k: int = 5,
                       visualize_top_n: int = 3,
                       min_matches: int = 15,
                       gap_ratio: float = 2.0,
                       use_cache: bool = True):
    """
    Complete test pipeline with NO_MATCH detection and caching support.
    """
    print("="*80)
    print(f"TESTING MATCHER: Query {query_id:05d} vs BBDD ({descriptor_type.upper()})")
    print("="*80)
    
    # Load query image
    print("\n1. Loading query image...")
    query_image, query_path = load_query_image(query_id, query_folder)
    print(f"   → Loaded: {query_path}")
    print(f"   → Shape: {query_image.shape}")
    
    # Display query image
    print("\n2. Displaying query image...")
    plt.figure(figsize=(10, 8))
    plt.imshow(query_image)
    plt.title(f"Query Image {query_id:05d}", fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Match against BBDD
    print("\n3. Matching against BBDD...")
    top_matches = match_query_to_bbdd(
        query_image,
        bbdd_folder=bbdd_folder,
        descriptor_type=descriptor_type,
        top_k=top_k,
        use_cache=use_cache
    )
    
    # Classify match confidence
    confidence_level, confidence_score = classify_match_confidence(
        top_matches, 
        min_matches=min_matches,
        gap_ratio=gap_ratio
    )
    
    # Display results
    print("\n4. Top Matches:")
    print("="*80)
    print(f"Match Confidence: {confidence_level} ({confidence_score:.2f})")
    
    if confidence_level == 'NO_MATCH':
        print("\n⚠️  NO VALID MATCH FOUND IN DATABASE")
        print("    This query likely has no correspondence in BBDD")
    
    print("\n" + "-"*80)
    print(f"{'Rank':<6} {'BBDD ID':<10} {'Matches':<10} {'Score':<12}")
    print("-"*80)
    
    for rank, (bbdd_id, num_matches, score) in enumerate(top_matches, 1):
        marker = "✓" if rank == 1 and confidence_level in ['HIGH', 'MEDIUM'] else " "
        print(f"{marker} {rank:<5} {bbdd_id:<10} {num_matches:<10} {score:<12.4f}")
    
    print("="*80)
    
    # Visualize only if confidence is not NO_MATCH
    if confidence_level != 'NO_MATCH' and visualize_top_n > 0:
        print(f"\n5. Visualizing top {visualize_top_n} matches...")
        
        # Initialize descriptor for visualization
        if descriptor_type.lower() == 'orb':
            descriptor = ORBDescriptor(n_features=500, scale_factor=1.2, n_levels=8)
            norm_type = cv2.NORM_HAMMING
        elif descriptor_type.lower() == 'daisy':
            descriptor = DaisyDescriptor(step=16, radius=15, rings=3, histograms=8, orientations=8)
            norm_type = cv2.NORM_L2
        elif descriptor_type.lower() == 'sift':
            descriptor = SIFTDescriptor(n_features=0)
            norm_type = cv2.NORM_L2
        
        matcher = DescriptorMatcher(
            matcher_type="BF",
            norm_type=norm_type,
            cross_check=False,
            ratio_test_threshold=0.75
        )
        
        # Get query descriptors
        query_kp, query_desc = descriptor.detect_and_compute(query_image)
        
        # Load descriptor cache for visualization
        descriptor_cache = precompute_bbdd_descriptors(bbdd_folder, descriptor_type, force_recompute=False)
        
        for rank, (bbdd_id, num_matches, score) in enumerate(top_matches[:visualize_top_n], 1):
            print(f"\n  Visualizing rank {rank}: BBDD {bbdd_id:05d} ({num_matches} matches)")
            
            # Load BBDD image
            bbdd_image, _ = load_image_from_bbdd(bbdd_id, bbdd_folder)
            
            # Get cached descriptors and reconstruct keypoints
            cached_data = descriptor_cache[bbdd_id]
            bbdd_kp = reconstruct_keypoints(cached_data['keypoints_data'])
            bbdd_desc = cached_data['descriptors']
            
            # Get matches
            matches = matcher.match(query_desc, bbdd_desc)
            
            # Visualize
            visualize_matches(
                query_image, query_kp,
                bbdd_image, bbdd_kp,
                matches,
                query_id, bbdd_id
            )
    elif confidence_level == 'NO_MATCH':
        print("\n5. Skipping visualization (no valid match found)")
    
    print("\n" + "="*80)
    print("TEST COMPLETE!")
    print("="*80)
    
    return top_matches, confidence_level, confidence_score

def test_multiple_queries(query_ids: list[int] = [0, 1, 2, 3, 4],
                         query_folder: str = "qsd1_w4",
                         bbdd_folder: str = "BBDD",
                         descriptor_type: str = "orb",
                         top_k: int = 5,
                         use_cache: bool = True):
    """
    Test matching for multiple query images with caching support.
    
    Args:
        query_ids: List of query image IDs to test
        query_folder: Path to query folder
        bbdd_folder: Path to BBDD folder
        descriptor_type: 'orb', 'daisy', or 'sift'
        top_k: Number of top matches to retrieve
        use_cache: If True, use pre-computed descriptors
    """
    print("="*80)
    print(f"BATCH TESTING: {len(query_ids)} queries with {descriptor_type.upper()}")
    print("="*80)
    
    # Pre-compute descriptors once for all queries
    if use_cache:
        print("\nPre-loading BBDD descriptors...")
        descriptor_cache = precompute_bbdd_descriptors(bbdd_folder, descriptor_type, force_recompute=False)
        print(f"✓ Cached {len(descriptor_cache)} descriptors ready for matching\n")
    
    results = []
    
    for query_id in query_ids:
        print(f"\n{'='*80}")
        print(f"Processing Query {query_id:05d}")
        print('='*80)
        
        # Load query image
        query_image, _ = load_query_image(query_id, query_folder)
        
        # Match
        top_matches = match_query_to_bbdd(
            query_image,
            bbdd_folder=bbdd_folder,
            descriptor_type=descriptor_type,
            top_k=top_k,
            use_cache=use_cache
        )
        
        # Display top match
        if top_matches:
            print(f"  Top match: BBDD {top_matches[0][0]:05d} ({top_matches[0][1]} matches)")
        
        results.append({
            'query_id': query_id,
            'top_matches': top_matches
        })
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH TEST SUMMARY")
    print("="*80)
    print(f"Total queries processed: {len(query_ids)}")
    print(f"\nTop-1 matches for each query:")
    for result in results:
        if result['top_matches']:
            top_match = result['top_matches'][0]
            print(f"  Query {result['query_id']:05d} → BBDD {top_match[0]:05d} ({top_match[1]} matches)")
        else:
            print(f"  Query {result['query_id']:05d} → No matches found")
    print("="*80)
    
    return results

if __name__ == "__main__":
    # OPTION 0: Pre-compute descriptors (run this once)
    # print("\n" + "="*80)
    # print("OPTION 0: Pre-computing BBDD descriptors")
    # print("="*80)
    # precompute_bbdd_descriptors(bbdd_folder="BBDD", descriptor_type="orb", force_recompute=True)
    # precompute_bbdd_descriptors(bbdd_folder="BBDD", descriptor_type="sift", force_recompute=True)
    # precompute_bbdd_descriptors(bbdd_folder="BBDD", descriptor_type="daisy", force_recompute=True)
    
    # OPTION 1: Test single query (detailed visualization) with caching
    print("\n" + "="*80)
    print("OPTION 1: Testing single query with ORB (using cache)")
    print("="*80)
    results = test_query_matching(
        query_id=2,
        query_folder="qsd1_w4",
        bbdd_folder="BBDD",
        descriptor_type="orb",
        top_k=5,
        visualize_top_n=3,
        use_cache=True  # Use pre-computed descriptors
    )
    
    # OPTION 2: Test multiple queries (batch processing) with caching
    # print("\n" + "="*80)
    # print("OPTION 2: Testing multiple queries with ORB (using cache)")
    # print("="*80)
    # batch_results = test_multiple_queries(
    #     query_ids=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    #     query_folder="qsd1_w4",
    #     bbdd_folder="BBDD",
    #     descriptor_type="orb",
    #     top_k=5,
    #     use_cache=True  # Use pre-computed descriptors
    # )
    
    # OPTION 3: Compare different descriptors (all from cache)
    # print("\n" + "="*80)
    # print("OPTION 3: Comparing descriptors on query 0 (all cached)")
    # print("="*80)
    # for desc_type in ['orb', 'sift', 'daisy']:
    #     print(f"\n{'='*80}")
    #     print(f"Testing with {desc_type.upper()}")
    #     print('='*80)
    #     test_query_matching(
    #         query_id=0,
    #         descriptor_type=desc_type,
    #         top_k=5,
    #         visualize_top_n=0,
    #         use_cache=True
    #     )