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
                    $"サイズ不一致:\n{Path.GetFileName(path)}\nbytes={bytes.Length}, expected={expectedBytes}\nW/H設定を確認してください。",
                    "Error", MessageBoxButton.OK, MessageBoxImage.Error);
                _timer.Stop();
                return;
            }

            // WriteableBitmap (Gray16)
            if (_wb == null || _wb.PixelWidth != w || _wb.PixelHeight != h || _wb.Format != PixelFormats.Gray16)
            {
                _wb = new WriteableBitmap(w, h, 96, 96, PixelFormats.Gray16, null);
                Img.Source = _wb;
            }

            // Gray16: stride = w * 2 bytes
            int stride = w * 2;

            // UIスレッドでCopyPixels（TimerはUIスレッドでTickする）
            _wb.Lock();
            try
            {
                _wb.WritePixels(new Int32Rect(0, 0, w, h), bytes, stride, 0);
                _wb.AddDirtyRect(new Int32Rect(0, 0, w, h));
            }
            finally
            {
                _wb.Unlock();
            }
        }

        private void UpdateStatus()
        {
            string play = _timer.IsEnabled ? "Playing" : "Stopped";
            string info = _rawFiles.Count > 0
                ? $"{play}  {_index + 1}/{_rawFiles.Count}  {Path.GetFileName(_rawFiles[_index])}"
                : $"{play}  (no files)";

            TxtStatus.Text = info;
        }
    }
}
