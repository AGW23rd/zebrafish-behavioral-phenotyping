# Shoal Lab project notes

- Keep video processing local. The FastAPI service must bind to loopback only; never send recording data to a remote service.
- The current tracker is experimental OpenCV foreground segmentation and single-object centroid association, not a validated fish detector or pose-estimation model. Keep UI labels and documentation precise about this scope and report pixels when no spatial calibration is provided.
- Preserve the existing React, TypeScript, Vite, and CSS approach. Run `npm run lint` and `npm run build` after app changes.
