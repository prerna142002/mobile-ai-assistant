import pandas as pd
from tabulate import tabulate

# Phone specifications
data = [
    {
        'Model': 'Google Pixel 8a',
        'Price': '$499',
        'Display': '6.1" FHD+ OLED, 90Hz',
        'Processor': 'Google Tensor G3',
        'RAM': '8GB',
        'Storage': '128GB',
        'Rear Camera': '64MP + 13MP',
        'Front Camera': '13MP',
        'Battery': '4,492mAh',
        'OS': 'Android 14',
        '5G': 'Yes',
        'Water Resistance': 'IP67'
    },
    {
        'Model': 'OnePlus 12R',
        'Price': '$499',
        'Display': '6.78" LTPO AMOLED, 120Hz',
        'Processor': 'Snapdragon 8 Gen 2',
        'RAM': '8GB/16GB',
        'Storage': '128GB/256GB',
        'Rear Camera': '50MP + 8MP + 2MP',
        'Front Camera': '16MP',
        'Battery': '5,500mAh',
        'OS': 'OxygenOS 14 (Android 14)',
        '5G': 'Yes',
        'Water Resistance': 'IP64'
    }
]

def compare_phones():
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display comparison
    print("\n📱 Phone Comparison: Google Pixel 8a vs OnePlus 12R\n")
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    print("\nKey Differences:")
    print("1. Display: 12R has larger 6.78"" LTPO AMOLED with 120Hz vs Pixel's 6.1"" 90Hz")
    print("2. Processor: 12R has Snapdragon 8 Gen 2 vs Pixel's Tensor G3")
    print("3. Battery: 12R has larger 5,500mAh vs Pixel's 4,492mAh")
    print("4. Camera: Pixel has better computational photography, 12R has more camera sensors")
    print("5. Software: Pixel offers cleaner Android experience with faster updates")

if __name__ == "__main__":
    compare_phones()
