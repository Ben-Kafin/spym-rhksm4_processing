import numpy as np
import matplotlib.pyplot as plt
from spym.io import rhksm4
from scipy.integrate import simpson

def dIdV_line(ifile, chunk_index=0, cmap='jet', scalefactor=2.75):
    # Load the file
    f = rhksm4.load(ifile)
    
    # Extract positional data from Page 0
    size = int(np.shape(f._pages[0].data)[0])  # Number of points along the position axis
    pos = np.array([f._pages[0].attrs['RHK_Xoffset'] + i * f._pages[0].attrs['RHK_Xscale'] for i in range(size)]) * 1.0e10  # Convert position to Å
    
    # Apply the scale factor and shift position values to start at 0
    pos = (pos / scalefactor) - (pos[0] / scalefactor)

    # Flip the position axis and ensure it starts at 0
    pos = pos[::-1] - pos[::-1][0]  # Reverse and shift to start from 0

    # Extract energy and LIA current data from Page 6
    npts = int(np.shape(f._pages[6].data)[0])  # Number of energy points
    energy = np.array([f._pages[6].attrs['RHK_Xoffset'] + i * f._pages[6].attrs['RHK_Xscale'] for i in range(npts)])  # Energy values
    
    LIAcurrent = np.array(f._pages[6].data)  # Shape (npts, size), transpose to align with energy-row format

    # Slicing the LIAcurrent array into chunks
    slices = []
    for i in range(LIAcurrent.shape[1] // size):  # Number of slices
        slices.append(LIAcurrent[:, i*size:(i+1)*size])  # Extract each slice

    # Check if chunk_index is valid
    if chunk_index < 0 or chunk_index >= len(slices):
        print(f"Error: chunk_index {chunk_index} is out of range. There are only {len(slices)} chunks.")
        return
    
    # Select the specific chunk
    selected_chunk = slices[chunk_index]

    # Normalize each column of the selected chunk
    normalized_chunk = np.zeros_like(selected_chunk)
    for col_index in range(selected_chunk.shape[1]):
        curve = selected_chunk[:, col_index]  # Extract the column
        curve = curve - np.min(curve)  # Ensure all values are positive
        
        area = simpson(curve, energy)  # Calculate the area under the curve using Simpson's rule
        if area > 0:
            normalized_curve = curve / area  # Normalize the curve while maintaining shape
            normalized_chunk[:, col_index] = normalized_curve * np.max(curve) / np.max(normalized_curve)  # Rescale back to original range
        else:
            normalized_chunk[:, col_index] = curve  # If the area is 0, keep the original curve

    # Reverse the order of curves position-wise (columns)
    normalized_chunk = normalized_chunk[:, ::-1]

    # Transpose the normalized and reversed chunk
    normalized_chunk = normalized_chunk.T
        
    # Reverse the order of curves energy-wise (columns)
    normalized_chunk = normalized_chunk[:, ::-1]

    # Plot the normalized chunk with the corrected axes
    plt.figure(figsize=(8, 6))
    plt.imshow(normalized_chunk, extent=[energy[-1], energy[0], pos[0], pos[-1]],  # Correctly flipped position axis
               aspect='auto', cmap=cmap)
    plt.colorbar(label="dIdV signal")
    plt.xlabel("Energy (eV)")
    plt.ylabel("Position (Å)")
    plt.title(f"dIdV")
    plt.show()
    



def dIdV_map(ifile, cmap='jet', scalefactor=2.75):
 # Load the file
    f = rhksm4.load(ifile)
    
    # Extract positional data from Page 0
    size = int(np.shape(f._pages[0].data)[0])  # Number of points along the position axis
    pos = np.array([f._pages[0].attrs['RHK_Xoffset'] + i * f._pages[0].attrs['RHK_Xscale'] for i in range(size)]) * 1.0e10  # Convert position to Å
    
    # Apply the scale factor and shift position values to start at 0
    pos = (pos / scalefactor) - (pos[0] / scalefactor)
    # Flip the position axis and ensure it starts at 0
    pos = pos[::-1] - pos[::-1][0]  # Reverse and shift to start from 0

    # Extract energy and LIA current data from Page 6
    npts = int(np.shape(f._pages[6].data)[0])  # Number of energy points
    energy = np.array([f._pages[6].attrs['RHK_Xoffset'] + i * f._pages[6].attrs['RHK_Xscale'] for i in range(npts)])  # Energy values
    
    LIAcurrent = np.array(f._pages[6].data)  # Shape (npts, number of positions)

    # Step 1: Slice LIA current column by column, normalize each column, and reassemble
    normalized_columns = []
    for i in range(LIAcurrent.shape[1]):  # Iterate over each position (column)
        curve = LIAcurrent[:, i]  # Extract the column (LIA values for all energy levels at one position)
        curve = curve - np.min(curve)  # Ensure all values are positive
        
        area = simpson(curve, energy)  # Calculate the area under the curve using Simpson's rule
        if area > 0:
            normalized_curve = curve / area  # Normalize the curve
            normalized_columns.append(normalized_curve * np.max(curve) / np.max(normalized_curve))  # Rescale back to original range
        else:
            normalized_columns.append(curve)  # If the area is 0, keep the original curve

    # Combine normalized columns back together
    normalized_data = np.array(normalized_columns).T  # Reassemble and transpose

    # Step 2: Slice the transposed data into columns, where each column is energy-specific positional data
    energy_columns = []
    for i in range(normalized_data.shape[0]):  # Iterate over each energy level (row)
        energy_columns.append(normalized_data[i, :])  # Extract each column corresponding to one energy level

    # Step 3: Reshape each column into a square array and plot individually
    for energy_idx, energy_data in enumerate(energy_columns):
        if len(energy_data) != size**2:  # Ensure the column has the correct number of elements
            print(f"Skipping energy {energy[energy_idx]:.2f} eV: size mismatch")
            continue

        square_chunk = energy_data.reshape((size, size))  # Reshape to size x size
        # Flip the chunk vertically to place (0, 0) in the bottom-left corner
        #square_chunk = np.flipud(square_chunk)

        # Plot the square chunk
        plt.figure(figsize=(8, 6))
        plt.imshow(square_chunk, extent=[pos[0], pos[-1], pos[0], pos[-1]], 
                   aspect='auto', cmap=cmap)
        plt.colorbar(label="LIA Current (normalized)")
        plt.xlabel("Position (Å)")
        plt.ylabel("Position (Å)")
        plt.title(f"dIdV Map for Energy {energy[energy_idx]:.2f} eV")
        plt.show()





#STILL EDITING NEED TO INCLUDE ZSCALE STUFF
#Error: No valid topography data found for the selected mode.

def topo(ifile, mode='both', cmap='viridis', scalefactor=1.0):
    """
    Display forward and/or reverse topography data from an SM4 file.

    Parameters:
        ifile (str): Path to the SM4 file.
        mode (str): Choose 'forward', 'reverse', or 'both'. Defaults to 'forward'.
        cmap (str): Colormap for height visualization. Defaults to 'viridis'.
        scalefactor (float): Scale factor for x and y data. Defaults to 1.0.
    """
    # Load the file
    f = rhksm4.load(ifile)
    
    # Determine the selected mode
    data = None
    title = None
    if mode == 'forward' and 4 in f._pages:
        data = np.array(f._pages[4].data)  # Forward topography
        title = "Topography (Forward)"
        attrs = f._pages[4].attrs
    elif mode == 'reverse' and 5 in f._pages:
        data = np.array(f._pages[5].data)  # Reverse topography
        title = "Topography (Reverse)"
        attrs = f._pages[5].attrs
    elif mode == 'both' and 4 in f._pages and 5 in f._pages:
        data_forward = np.array(f._pages[4].data)
        data_reverse = np.array(f._pages[5].data)
        attrs = f._pages[4].attrs  # Use attributes from page 2 (common for both)
    else:
        print("Error: No valid topography data found for the selected mode.")
        return

    # Extract scaling and units from attributes
    x_scale = attrs['RHK_Xscale']
    y_scale = attrs['RHK_Yscale']
    z_scale = attrs['RHK_Zscale']
    x_unit = attrs.get('RHK_Xunits', 'a.u.')  # Default to arbitrary units if units are not specified
    y_unit = attrs.get('RHK_Yunits', 'a.u.')
    z_unit = attrs.get('RHK_Zunits', 'a.u.')

    # Create x and y position arrays
    size_x = data.shape[1] if data is not None else data_forward.shape[1]
    size_y = data.shape[0] if data is not None else data_forward.shape[0]
    x_pos = np.array([attrs['RHK_Xoffset'] + i * x_scale for i in range(size_x)]) / scalefactor
    y_pos = np.array([attrs['RHK_Yoffset'] + i * y_scale for i in range(size_y)]) / scalefactor

    # Convert topography data (z-values) using z_scale
    if data is not None:
        data = data * z_scale
    else:
        data_forward = data_forward * z_scale
        data_reverse = data_reverse * z_scale

    # Plot the data
    if mode == 'forward' or mode == 'reverse':
        plt.figure(figsize=(8, 6))
        plt.imshow(
            data,
            extent=[x_pos[0], x_pos[-1], y_pos[0], y_pos[-1]],
            aspect='auto',
            cmap=cmap,
        )
        plt.colorbar(label=f"Height ({z_unit})")
        plt.xlabel(f"Position X ({x_unit})")
        plt.ylabel(f"Position Y ({y_unit})")
        plt.title(title)
        plt.show()

    elif mode == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Forward Topography
        axes[0].imshow(
            data_forward,
            extent=[x_pos[0], x_pos[-1], y_pos[0], y_pos[-1]],
            aspect='auto',
            cmap=cmap,
        )
        axes[0].set_title("Topography (Forward)")
        axes[0].set_xlabel(f"Position X ({x_unit})")
        axes[0].set_ylabel(f"Position Y ({y_unit})")
        axes[0].colorbar = plt.colorbar(axes[0].images[0], ax=axes[0], label=f"Height ({z_unit})")

        # Reverse Topography
        axes[1].imshow(
            data_reverse,
            extent=[x_pos[0], x_pos[-1], y_pos[0], y_pos[-1]],
            aspect='auto',
            cmap=cmap,
        )
        axes[1].set_title("Topography (Reverse)")
        axes[1].set_xlabel(f"Position X ({x_unit})")
        axes[1].set_ylabel(f"Position Y ({y_unit})")
        axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1], label=f"Height ({z_unit})")

        plt.tight_layout()
        plt.show()
