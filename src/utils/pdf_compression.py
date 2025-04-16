import os
import matplotlib.pyplot as plt
import subprocess
from pathlib import Path

def compress_pdf_with_ghostscript(input_file, output_file=None, quality='ebook'):
    """
    Compress a PDF file using Ghostscript.
    
    Parameters:
    ----------
    input_file : str
        Path to the input PDF file
    output_file : str, optional
        Path to the output PDF file. If None, will overwrite the input file.
    quality : str, optional
        Compression quality. Options are:
        - 'screen' (72 dpi, smallest files)
        - 'ebook' (150 dpi, good balance)
        - 'printer' (300 dpi, better quality)
        - 'prepress' (300 dpi, color preserving)
        - 'default' (equivalent to 'printer')
    
    Returns:
    -------
    bool
        True if successful, False otherwise
    """
    if output_file is None:
        # Create a temporary file and then replace the original
        temp_output = str(input_file) + '.temp.pdf'
    else:
        temp_output = output_file
    
    # Make sure the quality parameter is valid
    valid_qualities = ['screen', 'ebook', 'printer', 'prepress', 'default']
    if quality not in valid_qualities:
        quality = 'ebook'
    
    # Ghostscript command
    gs_command = [
        'gs',
        '-sDEVICE=pdfwrite',
        '-dCompatibilityLevel=1.4',
        '-dPDFSETTINGS=/' + quality,
        '-dNOPAUSE',
        '-dQUIET',
        '-dBATCH',
        f'-sOutputFile={temp_output}',
        input_file
    ]
    
    try:
        subprocess.run(gs_command, check=True)
        
        # If no output_file specified, replace the original
        if output_file is None:
            os.replace(temp_output, input_file)
        
        # Compare file sizes
        original_size = os.path.getsize(input_file if output_file else input_file)
        compressed_size = os.path.getsize(output_file if output_file else input_file)
        
        compression_ratio = (original_size - compressed_size) / original_size * 100
        print(f"PDF compressed: {original_size/1024/1024:.2f}MB → {compressed_size/1024/1024:.2f}MB ({compression_ratio:.1f}% reduction)")
        
        return True
    except subprocess.CalledProcessError:
        print("Failed to compress PDF with Ghostscript. Is Ghostscript installed?")
        if os.path.exists(temp_output) and output_file is None:
            os.remove(temp_output)
        return False
    except Exception as e:
        print(f"Error compressing PDF: {e}")
        if os.path.exists(temp_output) and output_file is None:
            os.remove(temp_output)
        return False

def compress_pdfs_in_directory(directory, pattern="*.pdf", quality='ebook'):
    """
    Compress all PDF files in a directory matching a pattern
    
    Parameters:
    ----------
    directory : str
        Directory containing PDF files
    pattern : str, optional
        Glob pattern to match PDF files
    quality : str, optional
        Compression quality (see compress_pdf_with_ghostscript)
    
    Returns:
    -------
    int
        Number of files successfully compressed
    """
    pdf_files = list(Path(directory).glob(pattern))
    success_count = 0
    
    print(f"Found {len(pdf_files)} PDF files in {directory}")
    
    for pdf_file in pdf_files:
        print(f"Compressing {pdf_file.name}...")
        if compress_pdf_with_ghostscript(str(pdf_file), quality=quality):
            success_count += 1
    
    print(f"Successfully compressed {success_count} out of {len(pdf_files)} PDF files")
    return success_count

def simplify_plot_for_pdf(plt, simplify_tolerance=0.001):
    """
    Reduce the complexity of plot lines to decrease PDF file size
    """
    import matplotlib as mpl
    
    # Store original settings
    original_simplify = mpl.rcParams['path.simplify']
    original_tolerance = mpl.rcParams['path.simplify_threshold']
    
    # Apply simplification settings
    mpl.rcParams['path.simplify'] = True
    mpl.rcParams['path.simplify_threshold'] = simplify_tolerance  # Higher means more simplification
    
    # Return a function to restore original settings
    def restore_settings():
        mpl.rcParams['path.simplify'] = original_simplify
        mpl.rcParams['path.simplify_threshold'] = original_tolerance
    
    return restore_settings

def save_optimized_pdfs(results_dir, file_pattern="*.pdf"):
    """
    Apply Ghostscript compression to all PDFs in the specified directory
    """
    # Check if Ghostscript is installed
    try:
        subprocess.run(['gs', '--version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        compress_pdfs_in_directory(results_dir, pattern=file_pattern, quality='ebook')
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Ghostscript not found. Please install Ghostscript for PDF compression.")
        print("On Ubuntu/Debian: sudo apt-get install ghostscript")
        print("On macOS: brew install ghostscript")
        print("On Windows: Download from https://ghostscript.com/releases/gsdnld.html")
        return False
    
    return True



# For batch compression of existing PDFs
if __name__ == "__main__":
    # Example usage - compress all PDFs in a results directory
    # Change this to your actual results directory
    results_dir = "./results"
    save_optimized_pdfs(results_dir)