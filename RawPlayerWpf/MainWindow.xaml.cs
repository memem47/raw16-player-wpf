using System;
using System.IO;
using System.Linq;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using System.Collections.Generic;
using Microsoft.Win32;

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
                Title = "RAWフォルダ内の任意のRAWファイルを選択してください",
                Filter = "RAW files (*.raw)|*.raw",
                CheckFileExists = true
            };

            if (dlg.ShowDialog() != true) return;

            // 選択したRAWファイルの「親フォルダ」を使う
            string folder = System.IO.Path.GetDirectoryName(dlg.FileName);
            LoadFolder(folder);
        }

        private void LoadFolder(string folder)
        {
            // frame_XXXX.raw などを想定してソート
            _rawFiles = Directory.GetFiles(folder, "*.raw")
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

            int w = _wb8.PixelWidth;
            int h = _wb8.PixelHeight;

            // WL/WW
            double wl = SldWL.Value;
            double ww = Math.Max(1.0, SldWW.Value);

            // Windowの下限/上限
            double low = wl - ww / 2.0;
            double high = wl + ww / 2.0;
            double inv = 255.0 / (high - low);

            int n = w * h;
            for (int i = 0; i < n; i++)
            {
                double v = _currentFrame16[i];

                if (v <= low) _display8[i] = 0;
                else if (v >= high) _display8[i] = 255;
                else _display8[i] = (byte)((v - low) * inv);
            }

            // Gray8: stride = w
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


    }
}
