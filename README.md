# Contagion

Shape detection by recursive pixel spreading. No OpenCV, no learned model:
just a binary image, a recursive "infection" that collects each shape pixel by
pixel, and a bit of statistics around each shape's barycentre.

![Contagion on a test sheet](examples/output.png)

```
square    at ( 680,  139)   25728 px  corners=4  cv=0.110
circle    at ( 140,  140)   20330 px  corners=0  cv=0.004
ellipse   at ( 405,  140)   21948 px  corners=2  cv=0.225
triangle  at ( 827,  329)    3247 px  corners=3  cv=0.233
triangle  at ( 155,  407)   12231 px  corners=3  cv=0.236
rectangle at ( 420,  384)   26346 px  corners=4  cv=0.278
circle    at ( 680,  370)   11497 px  corners=0  cv=0.004
```

The 120 specks of dust on the [input](examples/input.png) are gone: the sieve
dropped them.

## How it works

1. **Infinite contrast.** The image is pushed until every pixel is 0 or 1
   (Otsu's method picks the threshold unless you give one).
2. **Scan.** A loop walks the rows, pixel by pixel, until it lands on a 1.
3. **Contagion.** That pixel is recorded and set to 0, then a recursive
   function looks at every pixel within a radius *r*: each 1 it finds is
   recorded, set to 0, and spreads the infection from there. The recursion ends
   when everything around is 0. The whole shape is now one cluster, and it is
   gone from the image.
4. **Repeat.** The scan resumes where it stopped, until the image is all zeros.
5. **Sieve.** Clusters smaller than *X* pixels are discarded as noise.
6. **Classification from the barycentre.** For each cluster, take the distance
   from the barycentre to every contour pixel:
   - its **coefficient of variation** (standard deviation / mean) is near zero
     only for a circle;
   - plotted against the angle, that distance rises to **one peak per corner**:
     2 for an ellipse, 3 for a triangle, 4 for a quadrilateral, and comparing
     the dips between the 4 peaks tells a square from a rectangle.

   Because it is measured around the barycentre, none of this depends on how
   the shape is rotated.
7. **Overlay.** Each barycentre gets its label and its contour, drawn over the
   original image.

The radius *r* matters: `r = 1` spreads to the 8 touching pixels; a larger
radius jumps small gaps, so a dashed or slightly broken outline stays one shape.

## Run it

```bash
pip install -r requirements.txt
python demo.py                                  # draws the test sheet, writes examples/
python contagion.py photo.png -o out.png --radius 2 --min-pixels 100
pytest -q                                       # 16 tests
```

## Notes

- The contagion recurses once per pixel, so a large shape goes tens of
  thousands of calls deep. It runs in a worker thread with a large stack and a
  raised recursion limit; an explicit stack would do the same job iteratively,
  but the recursive version is the idea as I first wrote it.
- Pure Python on the scan: the 900×520 demo sheet takes about 0.2 s.
- Limits: shapes must not touch each other (they would be one cluster), and the
  thresholds in `contagion.py` are tuned for clean, filled shapes.

## Origin

I came up with this algorithm in 2022. This repository is a 2026
re-implementation of that design, written with Claude Code.

## License

MIT, see [LICENSE](LICENSE).
