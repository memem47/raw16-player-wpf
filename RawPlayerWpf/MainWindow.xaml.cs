using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Win32;

using ImageProcCli;

namespace RawPlayerWpf
{
    public partial class MainWindow : Window
    {
        private List<string> _rawFiles = new List<string>();
        private int _index = 0;

        private WriteableBitmap _wb;
        private DispatcherTimer _timer;

        private ushort[] _currentFrame16;   // 現在フレームのushort配列（WL/WW用に保持）
        private byte[] _display8;           // 表示用8bitバッファ（使い回し）
        private WriteableBitmap _wb8;       // Gray8表示用
        
        public MainWindow()
        {
            InitializeComponent();

            _timer = new DispatcherTimer(DispatcherPriority.Render);
            _timer.Tick += (_, __) => NextFrameInternal();
            UpdateStatus();
        }

        private void BtnSelectFolder_Click(object sender, RoutedEventArgs e)
        {
            var dlg = new OpenFileDialog
            {
                Title = "RAWフォルダ内の任意のRAW/BINファイルを選択してください",
                Filter = "RAW files (*.raw;*.bin)|*.raw;*.bin|RAW (*.raw)|*.raw|BIN (*.bin)|*.bin",
                CheckFileExists = true
            };

            if (dlg.ShowDialog() != true) return;

            // 選択したRAWファイルの「親フォルダ」を使う
            string folder = System.IO.Path.GetDirectoryName(dlg.FileName);
            LoadFolder(folder);
        }

        private void LoadFolder(string folder)
        {
            // 画像サイズ自動セット
            if (TryLoadMetaSize(folder, out int mw, out int mh) ||
                TryGuessSizeFromFirstRaw(folder, out mw, out mh))
            {
                TxtW.Text = mw.ToString();
                TxtH.Text = mh.ToString();
            }
            _rawFiles = Directory.GetFiles(folder)
                                 .Where(p =>
                                 {
                                     string ext = System.IO.Path.GetExtension(p).ToLowerInvariant();
                                     return ext == ".raw" || ext == ".bin";
                                 })
                                 .OrderBy(p => p, StringComparer.OrdinalIgnoreCase)
                                 .ToList();

            _index = 0;

            if (_rawFiles.Count == 0)
            {
                MessageBox.Show("RAWファイルが見つかりませんでした。", "Info", MessageBoxButton.OK, MessageBoxImage.Information);
                UpdateStatus();
                return;
            }

            // 初回表示
            ShowFrame(_index);
            UpdateStatus();
        }

        private bool TryLoadMetaSize(string folder, out int w, out int h)
        {
            w = h = 0;
            string metaPath = System.IO.Path.Combine(folder, "meta.json");
            if (!File.Exists(metaPath)) return false;

            try
            {
                var json = File.ReadAllText(metaPath);
                var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;

                if (root.TryGetProperty("width", out var jw) && root.TryGetProperty("height", out var jh))
                {
                    w = jw.GetInt32();
                    h = jh.GetInt32();
                    return (w > 0 && h > 0);
                }
            }
            catch { /* 読めなければ無視 */}

            return false;
        }

        private bool TryGuessSizeFromFirstRaw(string folder, out int w, out int h)
        {
            w = h = 0;
            var first = Directory.GetFiles(folder, "*.raw")
                .Concat(Directory.GetFiles(folder, "*.bin"))
                .OrderBy(p => p)
                .FirstOrDefault();
            if (first == null) return false;

            long bytes = new FileInfo(first).Length;
            if (bytes % 2 != 0) return false;

            long nPix = bytes / 2;

            //画像の縦横サイズが同じ
            int root = (int)Math.Sqrt(nPix);
            if (root * root == nPix)
            {
                w = h = root;
                return true;
            }

            // サイズ候補
            (int W, int H)[] candidates =
            {
                (640, 480),
                (768, 512),
                (800, 600),
                (1024, 768),
                (1280, 720),
                (1280, 1024),
                (1920, 1080),
            };

            foreach (var c in candidates)
            {
                if ((long)c.W * c.H == nPix)
                {
                    w = c.W; h = c.H;
                    return true;
                }
            }
            return false;
        }

        private void BtnPlay_Click(object sender, RoutedEventArgs e)
        {
            if (_rawFiles.Count == 0) return;

            if (!int.TryParse(TxtFps.Text, out int fps) || fps <= 0) fps = 30;

            // fps -> interval
            _timer.Interval = TimeSpan.FromMilliseconds(1000.0 / fps);
            _timer.Start();
            UpdateStatus();
        }

        private void BtnStop_Click(object sender, RoutedEventArgs e)
        {
            _timer.Stop();
            UpdateStatus();
        }

        private void BtnPrev_Click(object sender, RoutedEventArgs e)
        {
            if (_rawFiles.Count == 0) return;
            _timer.Stop();

            _index--;
            if (_index < 0) _index = ChkLoop.IsChecked == true ? _rawFiles.Count - 1 : 0;

            ShowFrame(_index);
            UpdateStatus();
        }

        private void BtnNext_Click(object sender, RoutedEventArgs e)
        {
            if (_rawFiles.Count == 0) return;
            _timer.Stop();

            NextFrameInternal();
            UpdateStatus();
        }

        private void NextFrameInternal()
        {
            if (_rawFiles.Count == 0) return;

            _index++;
            if (_index >= _rawFiles.Count)
            {
                if (ChkLoop.IsChecked == true) _index = 0;
                else { _timer.Stop(); _index = _rawFiles.Count - 1; }
            }

            ShowFrame(_index);
            UpdateStatus();
        }
        private void ShowFrame(int idx)
        {
            if (!int.TryParse(TxtW.Text, out int w) || w <= 0) w = 512;
            if (!int.TryParse(TxtH.Text, out int h) || h <= 0) h = 512;

            string path = _rawFiles[idx];

            int expectedBytes = w * h * 2;
            byte[] bytes = File.ReadAllBytes(path);

            if (bytes.Length != expectedBytes)
            {
                MessageBox.Show(
                    $"サイズ不一致:\n{System.IO.Path.GetFileName(path)}\nbytes={bytes.Length}, expected={expectedBytes}\nW/H設定を確認してください。",
                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                _timer.Stop();
                return;
            }

            // ushort配列へ（little-endian前提）
            int nPix = w * h;
            if (_currentFrame16 == null || _currentFrame16.Length != nPix)
                _currentFrame16 = new ushort[nPix];

            Buffer.BlockCopy(bytes, 0, _currentFrame16, 0, expectedBytes);

            // 表示用 WriteableBitmap(Gray8) を準備
            if (_wb8 == null || _wb8.PixelWidth != w || _wb8.PixelHeight != h || _wb8.Format != PixelFormats.Gray8)
            {
                _wb8 = new WriteableBitmap(w, h, 96, 96, PixelFormats.Gray8, null);
                Img.Source = _wb8;
            }

            if (_display8 == null || _display8.Length != nPix)
                _display8 = new byte[nPix];

            // WL/WWで表示を更新
            RenderCurrentFrameWithWlWw();
        }


        private void UpdateStatus()
        {
            string play = _timer.IsEnabled ? "Playing" : "Stopped";
            string info = _rawFiles.Count > 0
                ? $"{play}  {_index + 1}/{_rawFiles.Count}  {Path.GetFileName(_rawFiles[_index])}"
                : $"{play}  (no files)";

            TxtStatus.Text = info;
        }
        private void SldWL_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtWL != null)
                TxtWL.Text = ((int)SldWL.Value).ToString();
            RenderCurrentFrameWithWlWw();
        }

        private void SldWW_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (TxtWW != null)
                TxtWW.Text = ((int)SldWW.Value).ToString();
            RenderCurrentFrameWithWlWw();
        }

        private void BtnResetWlWw_Click(object sender, RoutedEventArgs e)
        {
            // とりあえず「それっぽい」初期値に戻す（必要なら後で自動推定に変える）
            SldWL.Value = 20000;
            SldWW.Value = 40000;
        }
        private void RenderCurrentFrameWithWlWw()
        {
            if (_currentFrame16 == null || _wb8 == null || _display8 == null) return;
            if (SldWL == null || SldWW == null) return;

            int w = _wb8.PixelWidth;
            int h = _wb8.PixelHeight;

            // まず表示用の入力は「現在フレーム」
            ushort[] src16 = _currentFrame16;

            // ===== ここから：CPU/GPU処理分岐 =====
            // UIがまだ初期化中の可能性もあるのでnullガード
            bool useCpu = (ChkDenoise != null && ChkDenoise.IsChecked == true);
            bool useGpu = (ChkDenoiseGpu != null && ChkDenoiseGpu.IsChecked == true);

            if (useCpu)
            {
                double ms;
                // C++/CLI CPU版
                ushort[] filtered = CpuFilters.Box3x3(src16, w, h, out ms);
                src16 = filtered;

                if (TxtProcMs != null) TxtProcMs.Text = $"CPU: {ms:F3} ms";
            }
            else if (useGpu)
            {
                double ms;
                // C++/CLI GPU版（CUDA end-to-end計測を返す想定）
                ushort[] filtered = CpuFilters.Box3x3Cuda(src16, w, h, out ms);
                src16 = filtered;

                if (TxtProcMs != null) TxtProcMs.Text = $"GPU(e2e): {ms:F3} ms";
            }
            else
            {
                if (TxtProcMs != null) TxtProcMs.Text = "";
            }
            // ===== ここまで：CPU/GPU処理分岐 =====

            // ===== WL/WWで src16 を Gray8 に落として表示 =====
            double wl = SldWL.Value;
            double ww = Math.Max(1.0, SldWW.Value);

            double low = wl - ww / 2.0;
            double high = wl + ww / 2.0;
            double inv = 255.0 / (high - low);

            int n = w * h;
            for (int i = 0; i < n; i++)
            {
                double v = src16[i];
                if (v <= low) _display8[i] = 0;
                else if (v >= high) _display8[i] = 255;
                else _display8[i] = (byte)((v - low) * inv);
            }

            int stride = w;

            _wb8.Lock();
            try
            {
                _wb8.WritePixels(new Int32Rect(0, 0, w, h), _display8, stride, 0);
                _wb8.AddDirtyRect(new Int32Rect(0, 0, w, h));
            }
            finally
            {
                _wb8.Unlock();
            }
        }

        private void ChkDenoise_Checked(object sender, RoutedEventArgs e)
        {
            // CPUをONにしたらGPUはOFF（排他）
            if (ChkDenoiseGpu != null) ChkDenoiseGpu.IsChecked = false;
            RenderCurrentFrameWithWlWw();
        }

        private void ChkDenoise_Unchecked(object sender, RoutedEventArgs e)
        {
            RenderCurrentFrameWithWlWw();
        }

        private void ChkDenoiseGpu_Checked(object sender, RoutedEventArgs e)
        {
            // GPUをONにしたらCPUはOFF（排他）
            if (ChkDenoise != null) ChkDenoise.IsChecked = false;
            RenderCurrentFrameWithWlWw();
        }

        private void ChkDenoiseGpu_Unchecked(object sender, RoutedEventArgs e)
        {
            RenderCurrentFrameWithWlWw();
        }

    }
}
