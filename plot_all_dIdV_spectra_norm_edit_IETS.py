import spym
import numpy as np
import matplotlib.pyplot as plt

def normalize_spectra(spectra):
    normalized_spectra = []
    areas = []
    for y in spectra.T:  # Transpose to iterate over spectra correctly
        area = np.trapz(y)  # Calculate the area under the curve using the trapezoidal rule
        if area != 0:
            areas.append(area)
            normalized_spectra.append(y / area)
    return np.array(normalized_spectra).T, areas  # Transpose back to original shape and return areas

def find_non_zero_range(x_values, spectra):
    non_zero_range = []
    for spectrum in spectra.T:
        non_zero_indices = np.where(~np.isclose(spectrum, 0))[0]
        if non_zero_indices.size > 0:
            range_min = min(x_values[non_zero_indices[0]], x_values[non_zero_indices[-1]])
            range_max = max(x_values[non_zero_indices[0]], x_values[non_zero_indices[-1]])
            non_zero_range.append((range_min, range_max))
    print(f'non-zero range: {non_zero_range}')        
    return non_zero_range

def calculate_area(x_values, y_values):
    return np.trapz(y_values, x_values)

def plot_dIdV_spectra(file_path, offset=0, normalize=False, plot_derivative=False):
    data = spym.io.rhksm4.load(file_path)
    
    # Iterate through pages to find LIA_Current data
    for page in data:
        if page.label == 'LIA_Current':
            x_values = page.coords[0][1]  # Extract the x-coordinate values
            spectra = page.data.astype(np.float64)  # Ensure data type is float64
            break
    else:
        raise ValueError("LIA_Current data not found in the file")
    
    if normalize:
        included_spectra = []
        excluded_spectra = []
        ignored_spectra = []
        for spectrum in spectra.T:
            if np.all(np.isclose(spectrum, 0)):  # Ignore spectra that are zero the entire time
                ignored_spectra.append(spectrum)
                continue
            if np.any(np.isclose(spectrum, 0)):  # Identify excluded spectrum
                excluded_spectra.append(spectrum)
            else:
                included_spectra.append(spectrum)
        
        included_spectra = np.array(included_spectra).T if included_spectra else np.array([])
        excluded_spectra = np.array(excluded_spectra).T if excluded_spectra else np.array([])

        
        if included_spectra.size > 0:
            included_spectra, areas = normalize_spectra(included_spectra)
            
            # Normalize excluded spectra after included spectra
            if excluded_spectra.size > 0:
                for i in range(excluded_spectra.shape[1]):
                    excluded_area = np.trapz(excluded_spectra[:, i])
                    if excluded_area != 0:
                        excluded_spectra[:, i] /= excluded_area
    
    # Calculate the average curve from included spectra
    valid_spectra = [s for s in included_spectra.T if not np.any(np.isclose(s, 0))]
    valid_spectra = np.array(valid_spectra).T if valid_spectra else np.array([])
    
    # Calculate non-zero ranges for excluded spectra
    non_zero_range = find_non_zero_range(x_values, excluded_spectra)

    if valid_spectra.size > 0:
        avg_y = np.mean(valid_spectra, axis=1)
        
        # Calculate the total area under the average curve
        total_avg_area = calculate_area(x_values, avg_y)
        print(f'Total average area: {total_avg_area}')
        
        # Calculate the area under the average curve only in the non-zero range
        non_zero_avg_areas = []
        for r in non_zero_range:
            print(f'Calculating area for range: {r}')  # Added print statement
            mask = (x_values >= r[0]) & (x_values <= r[1])
            if np.any(mask):
                non_zero_avg_area = calculate_area(x_values[mask], avg_y[mask])
                non_zero_avg_areas.append(non_zero_avg_area)
                print(f'Non-zero average area in range {r}: {non_zero_avg_area}')
        
        total_non_zero_avg_area = sum(non_zero_avg_areas)
        
        # Calculate the normalization factor for excluded spectra
        if total_avg_area != 0:
            norm_factor_for_excluded = total_non_zero_avg_area / total_avg_area
        else:
            norm_factor_for_excluded = 0
        
        print(f'Normalization factor for excluded spectra: {norm_factor_for_excluded}')
        
        # Apply normalization factor to excluded spectra
        if excluded_spectra.size > 0:
            for i in range(excluded_spectra.shape[1]):
                excluded_spectra[:, i] *= norm_factor_for_excluded
    
    # Calculate the derivative if requested
    if plot_derivative:
        avg_y_derivative = np.gradient(avg_y, x_values)
    
    # Plotting
    if valid_spectra.size > 0:
        fig, ax1 = plt.subplots()
        
        if plot_derivative:
            # Plot the average curve
            ax1.plot(x_values, avg_y, 'k--', label='Average')
            ax1.set_ylabel('dI/dV (A/V)' + (' (Normalized)' if normalize else ''))
            ax1.tick_params(axis='y')
            
            # Plot the derivative (IETS)
            ax2 = ax1.twinx()
            ax2.plot(x_values, avg_y_derivative, 'b--', label='IETS (D^2I/dV^2)', color='b')
            ax2.set_ylabel('IETS (d2I/dV2)', color='b')
            ax2.tick_params(axis='y', labelcolor='b')
        else:
            # Plot the average curve as the bottom-most curve
            ax1.plot(x_values, avg_y, 'k--', label='Average')
            
            # Plot each individual included spectrum above the average curve
            for i in range(included_spectra.shape[1]):
                ax1.plot(x_values, included_spectra[:, i] + (i + 1) * offset, label=f'Spectra {i + 1}')
            
            # Plot excluded spectra as the top-most curve
            if excluded_spectra.size > 0:
                for j in range(excluded_spectra.shape[1]):
                    ax1.plot(x_values, excluded_spectra[:, j] + (len(included_spectra.T) + j + 1) * offset, 'r', label=f'Excluded Spectra {j + 1}')
        
        # Set plot labels and legend
        ax1.set_xlabel('Bias Voltage (V)')
        ax1.set_title('dI/dV Spectra' + (' (Normalized)' if normalize else ''))
        ax1.legend(loc='upper left')  # Ensure legend is shown
    
        # Show the plot
        plt.show()
    else:
        raise ValueError("No valid spectra found for plotting")
