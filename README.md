# RAW16 Player (WPF .NET Framework 4.8)

WPF app to play sequential RAW16 (little-endian) grayscale frames.

## Requirements
- Visual Studio 2022
- .NET Framework 4.8

## RAW format
- uint16 little-endian
- size: width * height pixels
- row-major
- 1 frame bytes = width * height * 2

## How to run
1. Build and run from Visual Studio
2. Click "フォルダ選択" and select a `.raw` file in the target folder
3. Set Width/Height if needed
4. Press "再生"
