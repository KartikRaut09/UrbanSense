"""
Sentinel-2 Image Preprocessing Pipeline
----------------------------------------
Step 1: Download a Sentinel-2 tile from Copernicus Open Hub
        https://scihub.copernicus.eu  (free registration)
        → Search for your city → Download .SAFE folder

Step 2: Stack bands B04 (Red), B03 (Green), B02 (Blue) into a single GeoTIFF

Step 3: Slice large image into 256x256 training patches

Step 4: Generate binary masks from GeoJSON slum boundaries

Usage:
    # Stack bands
    python preprocess_sentinel2.py stack --safe_dir path/to/S2A.SAFE --output data/raw/stacked.tif

    # Tile the image
    python preprocess_sentinel2.py tile --input data/raw/stacked.tif --output_dir data/images/train/

    # Generate masks from GeoJSON
    python preprocess_sentinel2.py mask --geojson data/raw/slum_boundaries.geojson \
                                         --reference data/raw/stacked.tif \
                                         --output_dir data/masks/train/
"""

import argparse
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stack_sentinel2_bands(safe_dir: str, output_path: str):
    """
    Stack B04, B03, B02 bands from a Sentinel-2 .SAFE directory into one GeoTIFF.
    Requires: rasterio, glob
    """
    import rasterio
    from rasterio.merge import merge
    import glob

    safe_path = Path(safe_dir)

    # Sentinel-2 band file patterns
    band_patterns = {
        "B04": "*B04_10m.jp2",   # Red
        "B03": "*B03_10m.jp2",   # Green
        "B02": "*B02_10m.jp2",   # Blue
    }

    band_files = {}
    for band, pattern in band_patterns.items():
        matches = list(safe_path.rglob(pattern))
        if not matches:
            raise FileNotFoundError(f"Band {band} not found in {safe_dir}")
        band_files[band] = matches[0]
        logger.info(f"Found {band}: {matches[0]}")

    # Read and stack
    bands = []
    profile = None
    for band_name in ["B04", "B03", "B02"]:
        with rasterio.open(band_files[band_name]) as src:
            bands.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()

    stacked = np.stack(bands, axis=0)  # Shape: (3, H, W)
    profile.update(count=3, dtype="uint16")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)

    logger.info(f"✓ Stacked image saved to: {output_path}")
    logger.info(f"  Shape: {stacked.shape}, dtype: {stacked.dtype}")


def tile_image(input_path: str, output_dir: str, tile_size: int = 256, overlap: float = 0.5):
    """
    Slice a large GeoTIFF into 256x256 training patches.
    - overlap=0.5 → 50% overlap between patches (gives more training samples)
    - Skips mostly-dark patches (clouds, ocean, etc.)
    """
    import rasterio
    from rasterio.windows import Window

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = int(tile_size * (1 - overlap))  # Step size between patches
    count = 0
    skipped = 0

    with rasterio.open(input_path) as src:
        width, height = src.width, src.height
        logger.info(f"Image size: {width}x{height}")

        for row in range(0, height - tile_size, stride):
            for col in range(0, width - tile_size, stride):
                window = Window(col, row, tile_size, tile_size)
                data = src.read(window=window)

                # Skip mostly-black patches (no data areas)
                if np.mean(data) < 200:
                    skipped += 1
                    continue

                # Skip saturated patches (clouds)
                if np.mean(data) > 8000:
                    skipped += 1
                    continue

                profile = src.profile.copy()
                profile.update(
                    width=tile_size, height=tile_size,
                    transform=src.window_transform(window)
                )

                out_path = output_dir / f"patch_{count:05d}.tif"
                with rasterio.open(out_path, "w", **profile) as dst:
                    dst.write(data)
                count += 1

    logger.info(f"✓ Created {count} patches | Skipped {skipped} dark/cloud patches")
    logger.info(f"  Output directory: {output_dir}")


def generate_masks_from_geojson(geojson_path: str, reference_tif: str, output_dir: str):
    """
    Convert GeoJSON slum boundary polygons into binary GeoTIFF masks.

    The GeoJSON should contain Polygon/MultiPolygon features marking slum areas.
    Download from:
      - GRID3: https://grid3.org/resources/datasets
      - Map Kibera: https://mapkibera.org/wiki/
      - OSM Overpass: https://overpass-turbo.eu

    The output masks match the tile patches from tile_image().
    Each pixel: 1 = slum, 0 = non-slum
    """
    import rasterio
    from rasterio.features import rasterize
    import geopandas as gpd
    from rasterio.windows import Window

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load slum boundaries
    gdf = gpd.read_file(geojson_path)
    logger.info(f"Loaded {len(gdf)} slum polygons from GeoJSON")

    with rasterio.open(reference_tif) as ref:
        # Reproject GeoJSON to match satellite image CRS
        gdf = gdf.to_crs(ref.crs)

        # Get geometries as (geometry, value) tuples
        shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]

        # Rasterize the full image
        full_mask = rasterize(
            shapes=shapes,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=0,
            dtype=np.uint8
        )

    logger.info(f"Slum coverage: {np.mean(full_mask)*100:.2f}% of image")

    # Find corresponding image patches and create matching masks
    image_patches = sorted(Path(output_dir.parent.parent / "images" / output_dir.name).glob("*.tif"))

    if not image_patches:
        logger.warning("No image patches found to generate masks for. Run tile_image first.")
        # Save full mask as single file
        with rasterio.open(reference_tif) as ref:
            profile = ref.profile.copy()
            profile.update(count=1, dtype="uint8")
            with rasterio.open(output_dir / "full_mask.tif", "w", **profile) as dst:
                dst.write(full_mask, 1)
        return

    logger.info(f"Generating masks for {len(image_patches)} patches...")

    with rasterio.open(reference_tif) as ref:
        for patch_path in image_patches:
            with rasterio.open(patch_path) as patch_src:
                # Get pixel coordinates of this patch in the full image
                window = rasterio.windows.from_bounds(
                    *patch_src.bounds, transform=ref.transform
                )
                row_off = int(window.row_off)
                col_off = int(window.col_off)
                height = int(window.height)
                width = int(window.width)

                # Clip to image bounds
                row_off = max(0, min(row_off, full_mask.shape[0] - height))
                col_off = max(0, min(col_off, full_mask.shape[1] - width))

                patch_mask = full_mask[row_off:row_off+height, col_off:col_off+width]

                mask_path = output_dir / patch_path.name
                profile = patch_src.profile.copy()
                profile.update(count=1, dtype="uint8")
                with rasterio.open(mask_path, "w", **profile) as dst:
                    dst.write(patch_mask, 1)

    logger.info(f"✓ Saved {len(image_patches)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sentinel-2 Preprocessing Pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # Stack bands
    stack_parser = subparsers.add_parser("stack", help="Stack B04/B03/B02 bands")
    stack_parser.add_argument("--safe_dir", required=True, help="Path to .SAFE directory")
    stack_parser.add_argument("--output", required=True, help="Output stacked GeoTIFF path")

    # Tile image
    tile_parser = subparsers.add_parser("tile", help="Slice image into 256x256 patches")
    tile_parser.add_argument("--input", required=True, help="Input GeoTIFF path")
    tile_parser.add_argument("--output_dir", required=True, help="Output directory for patches")
    tile_parser.add_argument("--size", type=int, default=256, help="Tile size (default: 256)")
    tile_parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio (default: 0.5)")

    # Generate masks
    mask_parser = subparsers.add_parser("mask", help="Generate binary masks from GeoJSON")
    mask_parser.add_argument("--geojson", required=True, help="Path to slum GeoJSON file")
    mask_parser.add_argument("--reference", required=True, help="Reference GeoTIFF (for CRS/transform)")
    mask_parser.add_argument("--output_dir", required=True, help="Output directory for masks")

    args = parser.parse_args()

    if args.command == "stack":
        stack_sentinel2_bands(args.safe_dir, args.output)
    elif args.command == "tile":
        tile_image(args.input, args.output_dir, args.size, args.overlap)
    elif args.command == "mask":
        generate_masks_from_geojson(args.geojson, args.reference, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
