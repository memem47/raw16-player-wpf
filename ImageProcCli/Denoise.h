#pragma once

using namespace System;



namespace ImageProcCli
{
    public ref class CpuFilters sealed
    {
    public:
        // in/out は ushort[]（Gray16）
        // elapsedMs は DLL内で計測して返す
        static array<UInt16>^ Box3x3(array<UInt16>^ src, int width, int height, [Runtime::InteropServices::Out] double% elapsedMs);

        static array<UInt16>^ Box3x3Cuda(array<UInt16>^ src, int width, int height, [Runtime::InteropServices::Out] double% elapsedMs);
        // （任意）後で追加しやすいように：Medianも枠だけ用意してもよい
        // static array<UInt16>^ Median3x3(array<UInt16>^ src, int width, int height, [Out] double% elapsedMs);
    };
}
