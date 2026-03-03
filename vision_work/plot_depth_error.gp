# Gnuplot script for Kinect2 depth calibration error visualization
# Usage: gnuplot vision_work/plot_depth_error.gp

DATAFILE = "vision_work/018436651247/plot.dat"

# --- Plot 1: Error scatter map (X, Y pixel colored by error) ---
set terminal qt 0 title "Depth Error Heatmap" size 900,600
set title "Depth Calibration Error per Pixel"
set xlabel "X (pixel)"
set ylabel "Y (pixel)"
set cblabel "Error (m)"
set palette rgbformulae 33,13,10
set yrange [] reverse   # flip Y to match image coordinates
plot DATAFILE using 1:2:5 with points pt 7 ps 0.8 palette notitle

pause -1 "Press Enter for next plot..."

# --- Plot 2: Computed vs Measured depth ---
set terminal qt 1 title "Computed vs Measured Depth" size 700,600
set yrange [] noreverse
set title "Computed vs Measured Depth"
set xlabel "Computed depth (m)"
set ylabel "Measured depth (m)"
set key top left
plot DATAFILE using 3:4 with points pt 7 ps 0.3 lc rgb "#0077ff" title "data points", \
     x with lines lc rgb "red" lw 2 title "ideal (y=x)"

pause -1 "Press Enter for next plot..."

# --- Plot 3: Error histogram ---
set terminal qt 2 title "Error Distribution" size 700,500
set title "Depth Error Distribution"
set xlabel "Error (m)"
set ylabel "Count"
set yrange [0:]
binwidth = 0.002
bin(x, w) = w * floor(x/w + 0.5)
set boxwidth binwidth
set style fill solid 0.6
unset key
plot DATAFILE using (bin($5, binwidth)):(1) smooth frequency with boxes lc rgb "#e05000"

pause -1 "Press Enter to close..."
