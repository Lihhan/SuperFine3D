"""hfdetail: image-detail injection after the zoom refinement.
A fine-tuned, tiled normal estimator reads the input image; a height field on the input-image pixel grid moves the
visible vertices along the viewing direction so the rendered normals match. Works for any mesh in the object/dump
frame + any camera P (3x4, object -> normalized image coords), i.e. one crop or the whole stitched mesh at once.
Modules: camera, raster, visibility, targets, heightfield (reference + fast), estimator, pipeline."""
