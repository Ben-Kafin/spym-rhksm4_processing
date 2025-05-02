import numpy as np
import matplotlib.pyplot as plt
from spym.io import rhksm4
from scipy.signal import savgol_filter
from scipy.integrate import simpson

class DidVLineAnalyzer:
    def __init__(self, ifile, chunk_index=0, cmap='jet', scalefactor=2.75, combine_scans=False):
        """
        Initialize the dI/dV line analyzer.
          ifile         : file name (or path) to an SM4 file.
          chunk_index   : which full line (if more than one line is present)
          cmap          : colormap for plotting,
          scalefactor   : scale factor for converting positions
          combine_scans : if True, average the forward and backward scans
                          (when the scan is done in alternating directions)
        """
        self.ifile = ifile
        self.chunk_index = chunk_index
        self.cmap = cmap
        self.scalefactor = scalefactor
        self.combine_scans = combine_scans
        
        # Load the SM4 file and extract data
        self.f = rhksm4.load(self.ifile)
        self.extract_data()

    def extract_data(self):
        """Extract the position, energy, and dI/dV (LIAcurrent) data."""
        size = int(np.shape(self.f._pages[0].data)[0])
        pos = np.array([self.f._pages[0].attrs['RHK_Xoffset'] + i * self.f._pages[0].attrs['RHK_Xscale']
                        for i in range(size)]) * 1.0e10  # convert position to Å
        pos = (pos / self.scalefactor) - (pos[0] / self.scalefactor)
        # Reverse the positions and shift so that the first value is zero:
        self.pos = pos[::-1] - pos[::-1][0]

        npts = int(np.shape(self.f._pages[6].data)[0])
        self.energy = np.array([self.f._pages[6].attrs['RHK_Xoffset'] + i * self.f._pages[6].attrs['RHK_Xscale']
                               for i in range(npts)])
        self.LIAcurrent = np.array(self.f._pages[6].data)  # shape: (npts, total_scans)
        total_scans = self.LIAcurrent.shape[1]

        if total_scans == size:
            self.chunk = self.LIAcurrent.copy()
            self.scan_mode = 'single'
        elif total_scans % (2 * size) == 0:
            expected_scans = 2 * size
            n_chunks = total_scans // expected_scans
            if self.chunk_index < 0 or self.chunk_index >= n_chunks:
                raise ValueError(f"chunk_index {self.chunk_index} out of range. Only {n_chunks} chunks available.")
            chunk = self.LIAcurrent[:, self.chunk_index * expected_scans:(self.chunk_index + 1) * expected_scans]
            if self.combine_scans:
                forward = chunk[:, 0::2]
                backward = chunk[:, 1::2]
                self.chunk = (forward + backward) / 2.0
                self.scan_mode = 'combined'
            else:
                self.forward = chunk[:, 0::2]
                self.backward = chunk[:, 1::2]
                self.scan_mode = 'separate'
                self.chunk = None  # not used in separate mode
        else:
            # Fallback if the data dimensions don't match expectations
            self.chunk = self.LIAcurrent.copy()
            self.scan_mode = 'raw'

    def apply_hampel_filter(self, curves, window=3, n_sigma=2.5):
        """
        Apply a Hampel filter to each column (each spectrum) in curves.
          curves  : 2D array where each column is a spectrum.
          window  : Number of points on each side to include (window half-width).
          n_sigma : Threshold multiplier.
        Returns:
          Filtered curves (same shape as input).
        """
        filtered = np.zeros_like(curves)
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            current_curve = current_curve - np.min(current_curve)
            filtered_curve = current_curve.copy()
            k = int(window)
            for i in range(len(current_curve)):
                lower = max(0, i - k)
                upper = min(len(current_curve), i + k + 1)
                window_data = current_curve[lower:upper]
                med = np.median(window_data)
                mad = np.median(np.abs(window_data - med))
                threshold = n_sigma * 1.4826 * mad if mad != 0 else 0
                if np.abs(current_curve[i] - med) > threshold:
                    filtered_curve[i] = med
            filtered[:, j] = filtered_curve
        return filtered

    def apply_sgolay_filter(self, curves, window_length=5, polyorder=3):
        """
        Apply a Savitzky–Golay filter to each column (each spectrum) in curves.
          curves       : 2D array (npts x ncurves)
          window_length: Length of the filter window (must be odd)
          polyorder    : Order of the polynomial to fit
        Returns:
          Filtered curves (same shape as input)
        """
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(curves, window_length, polyorder, axis=0)

    def normalize_curves(self, curves):
        """
        Normalize each column (spectrum) so that its integrated area (via Simpson’s rule)
        is unity.
        """
        norm_curves = np.zeros_like(curves)
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            current_curve = current_curve - np.min(current_curve)
            area = simpson(current_curve, self.energy)
            if area > 0:
                norm_curves[:, j] = current_curve / area
            else:
                norm_curves[:, j] = current_curve
        return norm_curves

    def plot_line(self, hampel_filter_params=None, sgolay_filter_params=None,
                  apply_normalize=False):
        """
        Process and plot the dI/dV line scan.
        
        Optional filtering:
          hampel_filter_params: dictionary with keys 'window' and 'n_sigma'
          sgolay_filter_params: dictionary with keys 'window_length' and 'polyorder'
          apply_normalize      : if True, normalize each spectrum by its area.
          
        If both filters are provided they are applied sequentially (first Hampel, then SGolay).
        """
        self.extract_data()
        if self.scan_mode in ['single', 'combined', 'raw']:
            curves = self.chunk.copy()   # shape: (npts, N)
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            # Apply the same flipping operations as in your original code:
            curves = curves[:, ::-1]
            curves = curves.T
            curves = curves[:, ::-1]
            
            plt.figure(figsize=(8, 6))
            plt.imshow(curves, extent=[self.energy[-1], self.energy[0], self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV signal")
            if self.scan_mode == 'combined':
                plt.title("dI/dV (Combined Forward/Backward Scans)")
            else:
                plt.title("dI/dV Line Scan")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.show()
        elif self.scan_mode == 'separate':
            # Process forward and reverse scans separately.
            fwd = self.forward.copy()
            bck = self.backward.copy()
            if hampel_filter_params is not None:
                fwd = self.apply_hampel_filter(fwd, **hampel_filter_params)
                bck = self.apply_hampel_filter(bck, **hampel_filter_params)
            if sgolay_filter_params is not None:
                fwd = self.apply_sgolay_filter(fwd, **sgolay_filter_params)
                bck = self.apply_sgolay_filter(bck, **sgolay_filter_params)
            if apply_normalize:
                fwd = self.normalize_curves(fwd)
                bck = self.normalize_curves(bck)
            fwd = fwd[:, ::-1]
            fwd = fwd.T
            fwd = fwd[:, ::-1]
            bck = bck[:, ::-1]
            bck = bck.T
            bck = bck[:, ::-1]
            
            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(fwd, extent=[self.energy[-1], self.energy[0], self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Forward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Forward Scans")
            
            plt.subplot(1, 2, 2)
            plt.imshow(bck, extent=[self.energy[-1], self.energy[0], self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Backward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Backward Scans")
            plt.show()
        else:
            print("Unknown scan mode; cannot plot.")

# Sample run code:
if __name__ == '__main__':
    # Create an instance of DidVLineAnalyzer.
    # Change "example.sm4" to your actual SM4 file path.
    analyzer = DidVLineAnalyzer('C:/diectory/file.sm4', chunk_index=0, cmap='jet', scalefactor=2.75, combine_scans=False)
    
    # Plot with no filtering or normalization:
    analyzer.plot_line(apply_normalize=True)

    
    # Plot with Hampel filtering only:
    #analyzer.plot_line(hampel_filter_params={'window': 2.5, 'n_sigma': 2.5})
    
    # Plot with Savitzky–Golay filtering only:
    #analyzer.plot_line(sgolay_filter_params={'window_length': 5, 'polyorder': 3})
    
    # Plot with both filters applied sequentially (Hampel then SGolay) and then normalization:
    analyzer.plot_line(hampel_filter_params={'window': 3, 'n_sigma': 2.5},
                       sgolay_filter_params={'window_length': 5, 'polyorder': 3},
                       apply_normalize=True)
