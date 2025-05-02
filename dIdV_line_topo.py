# -*- coding: utf-8 -*-
"""
Created on Fri May  2 14:01:51 2025

@author: Benjamin Kafin
"""

import numpy as np
import matplotlib.pyplot as plt
from spym.io import rhksm4
from scipy.signal import savgol_filter
from scipy.integrate import simpson

class DidVLineAnalyzer:
    def __init__(self, ifile, chunk_index=0, cmap='jet', scalefactor=2.75, combine_scans=False):
        """
        Initialize the dI/dV line analyzer.
          ifile         : Path to an SM4 file.
          chunk_index   : Which full line (if more than one exists) for the dI/dV data.
          cmap          : Colormap for plotting.
          scalefactor   : Scale factor for converting positions.
          combine_scans : If True, average the forward and reverse (alternating) scans.
        """
        self.ifile = ifile
        self.chunk_index = chunk_index
        self.cmap = cmap
        self.scalefactor = scalefactor
        self.combine_scans = combine_scans

        # Load the SM4 file and extract dI/dV data.
        self.f = rhksm4.load(self.ifile)
        self.extract_data()

    def extract_data(self):
        """Extract the dI/dV (LIAcurrent) data, energy, and position."""
        size = int(np.shape(self.f._pages[0].data)[0])
        pos = np.array(
            [self.f._pages[0].attrs['RHK_Xoffset'] + i * self.f._pages[0].attrs['RHK_Xscale']
             for i in range(size)]
        ) * 1.0e10  # convert position to Å
        pos = (pos / self.scalefactor) - (pos[0] / self.scalefactor)
        # Reverse and shift so that the first value is zero.
        self.pos = pos[::-1] - pos[::-1][0]

        npts = int(np.shape(self.f._pages[6].data)[0])
        self.energy = np.array(
            [self.f._pages[6].attrs['RHK_Xoffset'] + i * self.f._pages[6].attrs['RHK_Xscale']
             for i in range(npts)]
        )
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
            chunk = self.LIAcurrent[:, self.chunk_index * expected_scans : (self.chunk_index + 1) * expected_scans]
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
            self.chunk = self.LIAcurrent.copy()
            self.scan_mode = 'raw'

    def apply_hampel_filter(self, curves, window=3, n_sigma=2.5):
        """Apply a Hampel filter to each column (spectrum) in curves."""
        filtered = np.zeros_like(curves)
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            # Remove baseline offset.
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
        """Apply a Savitzky–Golay filter to each column (spectrum) in curves."""
        if window_length % 2 == 0:
            window_length += 1
        return savgol_filter(curves, window_length, polyorder, axis=0)

    def normalize_curves(self, curves):
        """Normalize each spectrum (column) so its integrated area (via Simpson's rule) is 1."""
        norm_curves = np.zeros_like(curves)
        eps = 1e-8  # tolerance to avoid division by near-zero
        for j in range(curves.shape[1]):
            current_curve = curves[:, j]
            current_curve = current_curve - np.min(current_curve)
            area = simpson(current_curve, self.energy)
            if area > eps:
                norm_curves[:, j] = current_curve / area
            else:
                norm_curves[:, j] = current_curve
        return norm_curves

    def plot_line(self, hampel_filter_params=None, sgolay_filter_params=None, apply_normalize=False):
        """
        Process and plot the dI/dV line scan.
        Optional filtering and normalization are applied.
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
            # Follow the same flipping as in your original logic:
            curves = curves[:, ::-1]
            curves = curves.T
            curves = curves[:, ::-1]

            plt.figure(figsize=(8, 6))
            plt.imshow(curves, extent=[self.energy[-1], self.energy[0],
                                       self.pos[0], self.pos[-1]],
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
            plt.imshow(fwd, extent=[self.energy[-1], self.energy[0],
                                     self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Forward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Forward Scans")

            plt.subplot(1, 2, 2)
            plt.imshow(bck, extent=[self.energy[-1], self.energy[0],
                                     self.pos[0], self.pos[-1]],
                       aspect='auto', cmap=self.cmap)
            plt.colorbar(label="dI/dV (Backward)")
            plt.xlabel("Energy (eV)")
            plt.ylabel("Position (Å)")
            plt.title("dI/dV Backward Scans")
            plt.show()
        else:
            print("Unknown scan mode; cannot plot.")

    def extract_topography(self, topo_mode='forward', line_index=None):
        """
        Extract topography data from the SM4 file.
        
        Parameters:
          topo_mode: 'forward' -> use page 2,
                     'reverse' -> use page 3,
                     'average' -> average pages 2 and 3.
          line_index: (optional) if provided and the topo data is 2D,
                      this row is selected and returned as a 1D profile.
        
        Returns:
          If line_index is provided:
              (topo_x, topo_profile)
          Otherwise:
              (topo_x, topo_y, topo_data)
              
        The x-coordinates are computed from RHK_Xoffset and RHK_Xscale.
        The z-values (topography data) are scaled using the file attribute 'RHK_Zscale'.
        """
        if topo_mode == 'forward':
            page = self.f._pages[2]
        elif topo_mode == 'reverse':
            page = self.f._pages[3]
        elif topo_mode == 'average':
            if len(self.f._pages) < 4:
                raise ValueError("File does not contain both forward and reverse topo pages.")
            page_forward = self.f._pages[2]
            page_reverse = self.f._pages[3]
            data_forward = np.array(page_forward.data)
            data_reverse = np.array(page_reverse.data)
            topo_data = (data_forward + data_reverse) / 2.0
            page = page_forward
        else:
            raise ValueError("topo_mode must be 'forward', 'reverse', or 'average'.")

        # Determine the shape of the topography data.
        data_shape = np.shape(page.data)
        if len(data_shape) == 1:
            nx = data_shape[0]
            ny = 1
        else:
            ny, nx = data_shape

        # Compute coordinate arrays from the attributes.
        topo_x = np.array([page.attrs['RHK_Xoffset'] + i * page.attrs['RHK_Xscale']
                           for i in range(nx)]) * 1.0e10  # in Å
        topo_y = np.array([page.attrs['RHK_Yoffset'] + i * page.attrs['RHK_Yscale']
                           for i in range(ny)]) * 1.0e10
        topo_x = (topo_x / self.scalefactor) - (topo_x[0] / self.scalefactor)
        topo_y = (topo_y / self.scalefactor) - (topo_y[0] / self.scalefactor)

        # Get topo_data.
        if topo_mode in ['forward', 'reverse']:
            topo_data = np.array(page.data)
        # "average" already computed topo_data.

        # Use the file's Z scale.
        try:
            file_zscale = float(page.attrs['RHK_Zscale'])
        except KeyError:
            file_zscale = 1.0
        topo_data = (topo_data * file_zscale)*1E10

        # If line_index is provided, extract that row.
        if line_index is not None:
            if len(topo_data.shape) == 2:
                if line_index < 0 or line_index >= topo_data.shape[0]:
                    raise ValueError("line_index out of range.")
                topo_profile = topo_data[line_index, :]  # 1D array
                adjusted_topo = topo_profile-np.min(topo_profile)
                return topo_x, adjusted_topo
            else:
                # If data is not 2D, return as is.
                return topo_x, topo_data
        else:
            return topo_x, topo_y, topo_data

    def plot_line_with_topo(self, hampel_filter_params=None, sgolay_filter_params=None,
                            apply_normalize=False, topo_mode='forward', line_index=None):
        """
        Plot the dI/dV line scan alongside a topography profile.
        Parameters:
          hampel_filter_params: dict for the Hampel filter.
          sgolay_filter_params: dict for the Savitzky–Golay filter.
          apply_normalize      : If True, normalize each dI/dV spectrum.
          topo_mode            : 'forward' (use page 2), 'reverse' (use page 3) or 'average' (average pages 2 and 3).
          line_index           : If provided, this row from the topography data is used as the profile.
                                 (This is analogous to the chunk index in the spectroscopy data.)
        """
        # Process the dI/dV line scan data.
        self.extract_data()
        if self.scan_mode in ['single', 'combined', 'raw']:
            curves = self.chunk.copy()
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            curves = curves[:, ::-1]
        elif self.scan_mode == 'separate':
            curves = (self.forward + self.backward) / 2.0
            if hampel_filter_params is not None:
                curves = self.apply_hampel_filter(curves, **hampel_filter_params)
            if sgolay_filter_params is not None:
                curves = self.apply_sgolay_filter(curves, **sgolay_filter_params)
            if apply_normalize:
                curves = self.normalize_curves(curves)
            curves = curves[:, ::-1]
            curves = curves.T
            curves = curves[:, ::-1]
        else:
            raise ValueError("Unknown scan mode.")

        # Extract topography data.
        try:
            # Now, pass the line_index to extract_topography.
            if line_index is not None:
                topo_x, topo_profile = self.extract_topography(topo_mode, line_index=line_index)
            else:
                # If no line index is provided, default to central row.
                topo_x, topo_y, topo_data = self.extract_topography(topo_mode)
                topo_profile = topo_data[topo_data.shape[0] // 2, :]
        except Exception as e:
            print("Failed to extract topography:", e)
            return

        # Create a figure with two panels: left for dI/dV and right for the topo profile.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6),
                                       gridspec_kw={'width_ratios': [3, 1]})
        im = ax1.imshow(curves, extent=[self.energy[-1], self.energy[0],
                                         self.pos[0], self.pos[-1]],
                        aspect='auto', cmap=self.cmap)
        ax1.set_xlabel("Energy (eV)")
        ax1.set_ylabel("Position (Å)")
        if self.scan_mode == 'combined':
            ax1.set_title("dI/dV (Combined)")
        else:
            ax1.set_title("dI/dV Line Scan")
        plt.colorbar(im, ax=ax1, label="dI/dV signal")
        ax2.plot(topo_x, topo_profile, 'k-', linewidth=2)
        ax2.set_xlabel("Distance (Å)")
        ax2.set_ylabel("Height (Å)")
        ax2.set_title("Topography Profile (" + topo_mode + ")")
        ax2.set_xlim([np.min(topo_x), np.max(topo_x)])
        plt.tight_layout()
        plt.show()


# Sample run code:
if __name__ == '__main__':
    # Replace 'example.sm4' with the correct path.
    analyzer = DidVLineAnalyzer('C:/directory/file.sm4', chunk_index=0, cmap='jet', scalefactor=2.75, combine_scans=True)
    
    # Plot dI/dV line scan with filtering and normalization.
    analyzer.plot_line(hampel_filter_params={'window': 3, 'n_sigma': 2.5},
                       sgolay_filter_params={'window_length': 5, 'polyorder': 3},
                       apply_normalize=True)
    
    # Now plot dI/dV line scan together with topography.
    # Here we use topo_mode 'average' and explicitly choose a topography line (e.g., line_index=0).
    analyzer.plot_line_with_topo(hampel_filter_params={'window': 3, 'n_sigma': 2.5},
                                  sgolay_filter_params={'window_length': 5, 'polyorder': 3},
                                  apply_normalize=True,
                                  topo_mode='average',
                                  line_index=0)
