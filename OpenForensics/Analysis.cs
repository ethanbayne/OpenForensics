using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Threading.Tasks.Schedulers;

using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
using System.ComponentModel;
using System.Management;
using System.Collections.Concurrent;
using System.Linq;


namespace OpenForensics
{
    public partial class Analysis : Form
    {
        // Import possibility to gather raw access to devices
        // with global \\.\ paths which is prohibited by normal
        // .NET <cref>FileStream</cref> class.
        [DllImport("Kernel32.dll", SetLastError = true, CharSet = CharSet.Auto)]
        static extern SafeFileHandle CreateFile(
            string fileName,
            [MarshalAs(UnmanagedType.U4)] FileAccess fileAccess,
            [MarshalAs(UnmanagedType.U4)] FileShare fileShare,
            int securityAttributes,
            [MarshalAs(UnmanagedType.U4)] FileMode creationDisposition,
            [MarshalAs(UnmanagedType.U4)] FileAttributes fileAttributes,
            IntPtr template);

        public struct dataChunk
        {
            public int chunkNo;
            public byte[] chunkData;
            public bool peek;

            public dataChunk(int chunk, byte[] data)
            {
                chunkNo = chunk;
                chunkData = data;
                peek = false;
            }
        }

        public class dataReader
        {
            private FileStream DDStream;
            private static object locker = new Object();
            private ulong fileSize; 
            private int peek;
            private byte[] sectionEnd;
            private bool physicalDrive = false;
            private int readBuffer = 4 * 1024; //2 << 15; //32256;
            private int readQueues = 32;
            private int readLength = 1024 * 1024; //2 << 18;
            //private long position = 0;

            public dataReader(string path, int peekLength)
            {
                if (path.StartsWith("\\\\.\\"))
                {
                    physicalDrive = true;
                    // Set up Drive handle to read from physical device
                    SafeFileHandle driveHandle = CreateFile(@path, FileAccess.Read, FileShare.ReadWrite, 0, FileMode.Open, FileAttributes.Normal, IntPtr.Zero);
                    if (driveHandle.IsInvalid)
                        throw new Win32Exception(Marshal.GetLastWin32Error());
                    DDStream = new FileStream(driveHandle, FileAccess.Read);

                    // Get physical drive size
                    ManagementObjectSearcher mosDisks = new ManagementObjectSearcher("SELECT * FROM Win32_DiskDrive WHERE DeviceID = '" + path.Replace("\\", "\\\\") + "'");
                    foreach (ManagementObject moDisk in mosDisks.Get())
                        fileSize = (ulong)moDisk["Size"];

                    // Physical drive cannot handle partial sector jumps
                    //peek = 512;
                    peek = (int)((peekLength / 512) * 512);
                    if (peek == 0)
                        peek = 512;

                    //readBuffer = (100 * 1048576) / 32;
                    //readQueues = 32;

                    // Testing physical buffer overflows
                    //DDStream.Flush();
                    //DDStream.Position = 128031100000;
                    //DDStream.Position = 255052965888;

                    //MessageBox.Show(fileSize.ToString());
                }
                else
                {
                    DDStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None, readBuffer, FileOptions.SequentialScan);
                    fileSize = (ulong)DDStream.Length;

                    peek = peekLength;
                }

                sectionEnd = new byte[peek];
            }

            public int GetChunk(byte[] buffer, ref long count, ref double totalProcessed)
            {
                lock (locker)
                {
                    count = DDStream.Position;
                    int toRead = buffer.Length;
                    //MessageBox.Show(DDStream.IsAsync.ToString());

                    // Physical Drive Read Boundary Tests
                    if (physicalDrive && (ulong)count != fileSize && count > (long)(fileSize - 512))
                    {
                        DDStream.Flush();
                        DDStream.Position = (long)(fileSize - 512); // Make sure to read *full* disk, don't leave out any bytes at the end of the drive
                        count = DDStream.Position;
                        toRead = 512;
                    }
                    if ((ulong)(count + toRead) > fileSize)
                        toRead = (int)(fileSize - (ulong)count);

                    int result = 0;
                    if (toRead != 0)
                    {
                        int buffSplit = 0;
                        int oddBytes = 0;

                        if ((ulong)(count + toRead) != fileSize && count > 0 && toRead > peek)
                        {
                            toRead -= peek;
                            count -= peek;
                            Array.Copy(sectionEnd, buffer, peek);
                        }

                        if (toRead <= readLength)
                        {
                            buffSplit = 1;
                            oddBytes = toRead;
                        }
                        else
                        {
                            buffSplit = toRead / readLength;
                            oddBytes = toRead % readLength;
                            if (oddBytes != 0)
                                buffSplit++;
                        }

                        long currentPos = count;
                        ConcurrentQueue<int> queue = new ConcurrentQueue<int>();
                        for (int n = 0; n < buffSplit; n++)
                        {
                            int x = n;
                            int workLoad = readLength;
                            if (x == buffSplit - 1 && oddBytes != 0)
                                workLoad = oddBytes;
                            if (currentPos > 0)
                                queue.Enqueue(Process(DDStream, buffer, peek + (x * readLength), workLoad));
                            else
                                queue.Enqueue(Process(DDStream, buffer, x * readLength, workLoad));
                        }
                        
                        LimitedConcurrencyLevelTaskScheduler scheduler = new LimitedConcurrencyLevelTaskScheduler(readQueues);
                        TaskFactory factory = new TaskFactory(scheduler);

                        while (!queue.IsEmpty)
                        {
                            factory.StartNew(() =>
                            {
                                int task;
                                if (queue.TryDequeue(out task))
                                    result += task;

                            }, TaskCreationOptions.PreferFairness);
                        }

                        //for (int i = 0; i < buffSplit; i++)
                        //{
                        //    int task;
                        //    queue.TryDequeue(out task);
                        //    result += task;
                        //}

                        // Cache end of segment in memory to preserve DDStream cache
                        if (DDStream.Position != (long)fileSize)
                            Array.Copy(buffer, buffer.Length - peek, sectionEnd, 0, peek);
                    }

                    totalProcessed = DDStream.Position;
                    return result;
                }
            }

            private static int Process(FileStream fs, byte[] array, int start, int length)
            {
                //Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
                return fs.Read(array, start, length);
            }

            public int PastChunk(ref byte[] buffer, ref long count, long startLocation, int fileLength)
            {
                lock (locker)
                {
                    long rtnLocation = DDStream.Position;

                    DDStream.Flush();
                    DDStream.Position = startLocation;

                    count = DDStream.Position;
                    int readLength = DDStream.Read(buffer, 0, fileLength);

                    DDStream.Flush();
                    DDStream.Position = rtnLocation;

                    return readLength;
                }
            }

            public void CloseFile()
            {
                DDStream.Close();
            }

            public ulong GetFileSize()
            {
                return (ulong)fileSize;
            }
        }


        #region Form Load and Special Functions

        public Analysis()
        {
            InitializeComponent();
        }

        public class Input
        {
            public string TestType { get; set; }
            public bool carveOp { get; set; }
            public string GPGPU { get; set; }
            public List<int> gpus { get; set; }
            public long maxGPUMem { get; set; }
            public string FilePath { get; set; }
            public string CaseName { get; set; }
            public string EvidenceName { get; set; }
            public string saveLocation { get; set; }
            public List<string> targetName { get; set; }
            public List<string> targetHeader { get; set; }
            public List<string> targetFooter { get; set; }
        }

        private volatile bool shouldStop = false;

        private string TestType;
        private string GPGPU;
        private List<int> gpus;
        private long maxGPUMem;
        private int lpCount = Environment.ProcessorCount;
        private int gpuCoreCount;
        private int procShare;
        private string FilePath;
        private string CaseName;
        private string EvidenceName;
        private string saveLocation;
        private List<string> targetName;
        private List<string> targetHeader;
        private List<string> targetFooter;

        private Label[] gpuLabel;

        private uint chunkSize = 100 * 1048576;
        private int fileLength = 20 * 1048576;

        private Byte[][] target;
        private int[][] lookup;
        private Byte[][] targetEnd;
        private int[][] lookupEnd;

        private int longestTarget;

        private bool carveOp;

        private Stopwatch watch;
        private int[] results;
        private int[][] rewindCheck;
        private double totalProcessed;
        private uint chunkCount;
        private ConcurrentBag<String> foundResults = new ConcurrentBag<String>();
        //private List<Line> foundLocs = new List<Line>();
        private static object resultLock = new Object();

        private List<Engine> GPUCollection = new List<Engine>();

        //private int procThreads = int.MaxValue - 1;
        //private int[] activeThreads;// = int.MaxValue - 1;

        //private byte[][] tempbuffer;
        //private long[] tempcount;
        //private AutoResetEvent[] carveDone;

        public Input InputSet
        {
            set
            {
                TestType = value.TestType;
                carveOp = value.carveOp;
                GPGPU = value.GPGPU;
                gpus = value.gpus;
                maxGPUMem = value.maxGPUMem;
                FilePath = value.FilePath;
                CaseName = value.CaseName;
                EvidenceName = value.EvidenceName;
                saveLocation = value.saveLocation;
                targetName = value.targetName;
                targetHeader = value.targetHeader;
                targetFooter = value.targetFooter;
            }
        }

        private void Carve_Load(object sender, EventArgs e)
        {
            try
            {
                string techUsed = "";
                if (TestType == "CPU")
                    techUsed = " (" + lpCount + " Logical Cores)";
                else
                {
                    int maxGPUThread = (int)((maxGPUMem*0.8) / (chunkSize*2));
                    gpuCoreCount = Math.Min(lpCount / gpus.Count, maxGPUThread);
                    //gpuCoreCount = 4;

                    if (gpus.Count > 1)
                        techUsed = " (" + GPGPU + " - " + gpus.Count + " GPUs - running " + gpuCoreCount + " threads each)";
                    else
                        techUsed = " (" + GPGPU + " - running " + gpuCoreCount + " threads)";
                }

                //string pathType = "";
                //if (FilePath.StartsWith("\\\\.\\"))
                //    pathType = "Drive";
                //else
                //    pathType = "File";

                if (CaseName == "OpenForensics Output")
                    CaseName = "...";

                String[] startLog = {"OpenForensics Analysis Report",
                                    "",
                                    "Report Generated: " + String.Format("{0:HH:mm dd-MMM-yyyy}", DateTime.Now),
                                    "---------------------------------------------",
                                    "Case Reference: " + CaseName,
                                    "Evidence Reference: " + EvidenceName,
                                    "Technology Used: " + TestType + techUsed,
                                    "---------------------------------------------\n"};

                while (true)
                {
                    try
                    {
                        System.IO.File.WriteAllLines(saveLocation + "LogFile.txt", startLog);
                        break;
                    }
                    catch
                    {
                        Thread.Sleep(100);
                    }
                }

                DrawGPUStatus();

                if (!carveOp)
                    lblFound.Text = "Targets found:";

                Thread analysis = new Thread(new ThreadStart(FileAnalysis));
                analysis.Start();
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Carving Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private double MeasureTime(Action action)
        {
            // Simple stopwatch function for measuring processing time
            watch = new Stopwatch();

            watch.Start();
            action.Invoke();
            watch.Stop();

            return watch.ElapsedTicks / (double)Stopwatch.Frequency;
        }

        #endregion


        #region Interface Updates

        private void DrawGPUStatus()
        {
            if (TestType == "CPU")
            {
                grpGPUActivity.Text = "CPU Core Activity";

                gpuLabel = new Label[lpCount];
                for (int i = 0; i < lpCount; i++)
                    gpuLabel[i] = new Label() { Text = (i + 1).ToString(), AutoSize = false, Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter, Font = new System.Drawing.Font("Segoe UI", (float)(6.75)), BackColor = System.Drawing.Color.DimGray, BorderStyle = BorderStyle.FixedSingle, Margin = new Padding(0), Padding = new Padding(0) };

            }
            else
            {
                gpuLabel = new Label[gpuCoreCount * gpus.Count];
                for (int i = 0; i < gpuCoreCount * gpus.Count; i++)
                    gpuLabel[i] = new Label() { Text = (i + 1).ToString(), AutoSize = false, Dock = DockStyle.Fill, TextAlign = System.Drawing.ContentAlignment.MiddleCenter, Font = new System.Drawing.Font("Segoe UI", (float)(6.75)), BackColor = System.Drawing.Color.DimGray, BorderStyle = BorderStyle.FixedSingle, Margin = new Padding(0), Padding = new Padding(0) };
            }

            int gpuCount = gpuLabel.Length;
            int gpuPerRow = gpuCount;
            int rows = 1;

            //if(gpuCount > 12)
            //    rows = 3;
            if (gpuCount > 4)
                rows = 2;
            if (rows > 1)
                gpuPerRow = gpuCount / rows;

            tblGPU.ColumnCount = gpuPerRow;
            tblGPU.RowCount = rows;

            TableLayoutColumnStyleCollection colStyle = tblGPU.ColumnStyles;
            int colWidth = (tblGPU.Size.Width / gpuPerRow);
            colStyle[0].Width = colWidth;
            for (int i = 1; i < gpuLabel.Length; i++)
                tblGPU.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, colWidth));

            TableLayoutRowStyleCollection rowStyle = tblGPU.RowStyles;
            int rowHeight = (tblGPU.Size.Height / rows);
            rowStyle[0].Height = rowHeight;
            if (rows == 2)
                rowStyle[1].Height = rowHeight;

            for (int i = 0; i < gpuLabel.Length; i++)                
                tblGPU.Controls.Add(gpuLabel[i], i, 0);
        }


        private void updateHeader(string text)
        {
            try
            {
                if (this.lblHeader.InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        this.lblHeader.Text = text;
                    });
                }
                else
                {
                    this.lblHeader.Text = text;
                }
            }
            catch (Exception)
            { }
        }

        private void updateSegments()
        {
            try
            {
                if (this.lblSegments.InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        this.lblSegmentsValue.Text = chunkCount.ToString();
                    });
                }
                else
                {
                    this.lblSegmentsValue.Text = chunkCount.ToString();
                }
            }
            catch (Exception)
            { }
        }

        private void updateFound()
        {
            int total = 0;
            foreach (int count in results)
                total += count;

            try
            {
                if (this.lblFoundValue.InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                            this.lblFoundValue.Text = total.ToString();
                    });
                }
                else
                        this.lblFoundValue.Text = total.ToString();
            }
            catch (Exception)
            { }
        }

        private void updateGPUAct(int gpu, bool status, bool carving)
        {
            try
            {
                if (this.gpuLabel[gpu].InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        if(status)
                            if(carving)
                                this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightBlue;
                            else
                                this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightGreen;
                        else
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.Green;
                    });
                }
                else
                {
                    if (status)
                        if (carving)
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightBlue;
                        else
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightGreen;
                    else
                        this.gpuLabel[gpu].BackColor = System.Drawing.Color.Green;
                }
            }
            catch (Exception)
            { }
        }

        private void updateGPUAct(int gpu, bool finished)
        {
            try
            {
                if (finished)
                {
                    if (this.gpuLabel[gpu].InvokeRequired)
                    {
                        this.Invoke((MethodInvoker)delegate
                        {
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.DimGray;
                        });
                    }
                    else
                    {
                        this.gpuLabel[gpu].BackColor = System.Drawing.Color.DimGray;
                    }
                }
            }
            catch (Exception)
            { }
        }

        private void updateProgress(int percent, double position, double total)
        {
            try
            {
                double timeElapsed = (int)Math.Round(watch.Elapsed.TotalSeconds, 2);
                TimeSpan timeElapsedSpan = new TimeSpan(0, 0, (int)Math.Round(watch.Elapsed.TotalSeconds, 2));
                //double timeRemaining = (timeElapsed / percent) * (100 - percent);
                TimeSpan timeRemainingSpan = new TimeSpan(0, 0, (int)((timeElapsed / percent) * (100 - percent)));

                string formattedElapsed = string.Format("{0:00}:{1:00}:{2:00}", timeElapsedSpan.Hours, timeElapsedSpan.Minutes, timeElapsedSpan.Seconds);
                //string formattedElapsed = string.Format("{0:00}:{1:00}:{2:00}", timeElapsed / 3600, (timeElapsed / 60) % 60, timeElapsed % 60);
                string formattedRemaining = "--:--:--";
                if (percent > 0)
                    formattedRemaining = string.Format("{0:00}:{1:00}:{2:00}", timeRemainingSpan.Hours, timeRemainingSpan.Minutes, timeRemainingSpan.Seconds);
                //formattedRemaining = string.Format("{0:00}:{1:00}:{2:00}", timeRemaining / 3600, (timeRemaining / 60) % 60, timeRemaining % 60);

                if (this.pbProgress.InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        this.pbProgress.Value = percent;
                        this.lblProgress.Text = percent + "%";
                        this.lblProcess.Text = "Processing: " + position + " / " + total;
                        this.lblTimeElapsedValue.Text = formattedElapsed;
                        this.lblTimeRemainingValue.Text = formattedRemaining;
                    });
                }
                else
                {
                    this.pbProgress.Value = percent;
                    this.lblProgress.Text = percent + "%";
                    this.lblProcess.Text = "Processing: " + position + " / " + total;
                    this.lblTimeElapsedValue.Text = formattedElapsed;
                    this.lblTimeRemainingValue.Text = formattedRemaining;
                }              
            }
            catch (Exception)
            { }
        }

        #endregion


        #region Analysis Command

        private void FileAnalysis()
        {

            if (TestType == "CPU")
            {
                #region DD Carve - CPU Section

                if (carveOp)
                    updateHeader("Carving Started using " + TestType + " (Cores: " + lpCount + ")");
                else
                    updateHeader("Analysis Started using " + TestType + " (Cores: " + lpCount + ")");

                // Uncomment for single-CPU operation
                //lpCount = 1;

                //procThreads = 0;
                //activeThreads = new int[lpCount];
                //for (int i = 0; i < activeThreads.Length; i++)
                //    activeThreads[i] = 0;

                String path = FilePath;                                 // Target DD File

                LookupTableGen(true);

                if(carveOp)
                    results = new int[target.Length/2];           // Number of Results Found
                else
                    results = new int[target.Length];
                totalProcessed = 0;
                chunkCount = 0;                                 // Number of segments processed

                for (int i = 0; i < target.Length; i++)
                    if (target[i].Length > longestTarget)
                        longestTarget = target[i].Length;

                rewindCheck = new int[lpCount][];

                procShare = 1;

                //carveDone = new AutoResetEvent[lpCount];

                //FileInfo inputFile = new FileInfo(FilePath);
                //ulong fileSize = inputFile.Length;

                dataReader dataRead;
                if (carveOp)
                    dataRead = new dataReader(FilePath, fileLength);
                else
                    dataRead = new dataReader(FilePath, longestTarget - 1);

                ulong fileSize = dataRead.GetFileSize();

                // Start stopwatch, open file defined by user
                double time = MeasureTime(() =>
                {
                    Parallel.For(0, lpCount, i =>
                    {
                        CPUThread(i, ref dataRead);
                    });
                    Task.WaitAll();
                });

                dataRead.CloseFile();

                PrepareResults(fileSize, time);

                #endregion
            }
            else
            {
                #region DD Carve - GPU Section
                string gpuText = "";
                if (gpus.Count > 1)
                    gpuText = GPGPU + " (GPUs: " + gpus.Count + ")";
                else
                    gpuText = GPGPU;

                if (carveOp)
                    updateHeader("Carving Started using " + gpuText);
                else
                    updateHeader("Analysis Started using " + gpuText);

                LookupTableGen(true);

                if(carveOp)
                    results = new int[target.Length / 2];                  // Number of Results Found
                else
                    results = new int[target.Length]; 
                totalProcessed = 0;
                chunkCount = 0;                                 // Number of segments processed

                for (int i = 0; i < target.Length; i++)
                    if (target[i].Length > longestTarget)
                        longestTarget = target[i].Length;

                // Set GPU device as one selected in interface
                //CudafyModes.Target = eGPUType.Emulator;           // Set Target to OpenCL
                foreach(int GPUid in gpus)
                {
                    Engine gpu = new Engine(GPUid, gpuCoreCount, target, lookup, longestTarget, fileLength, chunkSize, carveOp);
                    GPUCollection.Add(gpu);
                }

                //procThreads = 0;
                //activeThreads = new int[GPUCollection.Count];
                //for (int i = 0; i < activeThreads.Length; i++)
                //    activeThreads[i] = 0;

                procShare = Math.Min(Math.Max(lpCount / gpuCoreCount / GPUCollection.Count, 1), lpCount);
                //procShare = 1;
                //MessageBox.Show(procShare.ToString());

                //tempbuffer = new byte[GPUCollection.Count][];
                //tempcount = new long[GPUCollection.Count];
                //rewindCheck = new int[GPUCollection.Count][];

                //carveDone = new AutoResetEvent[GPUCollection.Count];

                //FileInfo inputFile = new FileInfo(FilePath);
                //long fileSize = inputFile.Length;

                dataReader dataRead;
                if (carveOp)
                    dataRead = new dataReader(FilePath, fileLength);
                else
                    dataRead = new dataReader(FilePath, longestTarget - 1);

                ulong fileSize = dataRead.GetFileSize();

                // Start stopwatch, open file defined by user
                double time = MeasureTime(() =>
                {
                    Parallel.For(0, GPUCollection.Count, i =>
                    {
                        Parallel.For(0, gpuCoreCount, j =>
                        {
                            GPUThread(i, j, ref dataRead);
                        });
                    });
                    Task.WaitAll();
                });

                for (int i = 0; i < GPUCollection.Count; i++)
                {
                    GPUCollection[i].FreeAll();                                   // Free all GPU resources
                    GPUCollection[i].HostFreeAll();
                }
                dataRead.CloseFile();

                PrepareResults(fileSize, time);

                #endregion
            }
        }

        #region Result Prep

        private void PrepareResults(ulong fileSize, double time)
        {
            if (this.lblProcess.InvokeRequired)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    this.lblProcess.Text = "Finalising Log File...";
                });
            }
            else
            {
                this.lblProcess.Text = "Finalising Log File...";
            }

            try
            {
                using (System.IO.StreamWriter file = new System.IO.StreamWriter(saveLocation + "LogFile.txt", true))
                {
                    int duplicates = 0;
                    List<string> unique = new List<string>();
                    int[] resultsNew = new int[results.Length];
                    if (foundResults.Count > 0)
                    {
                        string[] foundLocs = foundResults.ToArray();
                        foreach (var item in foundLocs.OrderBy(x => double.Parse(x.Substring(0, x.IndexOf(' ')))))
                        {
                            int index = unique.FindIndex(x => x.StartsWith(item.Substring(0, item.IndexOf(' '))));
                        //if (!unique.Contains(item.Substring(0, item.IndexOf(' '))))
                            if(index == -1)
                            {
                                unique.Add(item);
                            
                                int nameIndex = targetName.IndexOf(item.Substring(item.LastIndexOf(' ') + 1, item.Length - item.LastIndexOf(' ') - 1));
                                resultsNew[nameIndex]++;
                            }
                            else if (unique[index].Contains("partial") && unique[index].Contains("fragmented") || !item.Contains("partial") && !item.Contains("fragmented"))
                            {
                                unique.RemoveAt(index);
                                unique.Add(item);
                            }
                        }
                        duplicates = foundResults.Count - unique.Count;
                    }

                    file.WriteLine("Statistics:");
                    file.WriteLine("-----------------------------");
                    file.WriteLine("Bytes Analysed: " + fileSize + " (" + Math.Round((double)fileSize / 1000000000,2).ToString() + " GB)");
                    file.WriteLine("File Segments : " + chunkCount);
                    file.WriteLine("");
                    if (shouldStop)
                    {
                        file.WriteLine("ANALYSIS ABORTED EARLY BY USER - RESULTS INCOMPLETE");
                        file.WriteLine("");
                    }
                    int total = 0;

                    if (carveOp)
                    {
                        foreach (int count in resultsNew)
                            total += count;
                        file.WriteLine("Total Files Carved: " + total);
                    }
                    else
                    {
                        foreach (int count in results)
                            total += count;
                        file.WriteLine("Total Targets Identified: " + total);
                    }
                    file.WriteLine("");


                    string timeInfo = "";
                    if (Math.Round(time, 2) >= 60)
                    {
                        TimeSpan timeElapsedSpan = new TimeSpan(0, 0, (int)Math.Round(watch.Elapsed.TotalSeconds, 2));
                        timeInfo = string.Format(" ({0:00}:{1:00}:{2:00})", timeElapsedSpan.Hours, timeElapsedSpan.Minutes, timeElapsedSpan.Seconds);
                    }

                    file.WriteLine("Processing Time: " + Math.Round(time, 2).ToString() + " seconds" + timeInfo);
                    file.WriteLine("Processing Rate: " + Math.Round(((chunkCount*(chunkSize/1048576)) / time),2).ToString() + " MB/sec");
                    file.WriteLine("-----------------------------\r\n");
                    file.WriteLine("");
                    file.WriteLine("Result Breakdown:");
                    file.WriteLine("-----------------------------");

                    //if (carveOp)
                    //{
                    //    if (results.Length > 1)
                    //    {
                    //        for (int i = 1; i < results.Length; i++)
                    //        {
                    //            if (targetName[i] == targetName[i - 1])
                    //            {
                    //                results[i - 1] += results[i];

                    //                int[] newStrItems = new int[results.Length - 1];
                    //                for (int j = 0, k = 0; j < newStrItems.Length; j++, k++)
                    //                {
                    //                    if (j == i)
                    //                        k++;

                    //                    newStrItems[j] = results[k];
                    //                }
                    //                results = newStrItems;

                    //                targetName.RemoveAt(i);

                    //                i -= 1;
                    //            }
                    //        }
                    //    }
                    //}

                    if (carveOp)
                    {
                        for (int i = 0; i < resultsNew.Length; i++)
                        {
                            file.WriteLine("Target: " + targetName[i]);
                            file.WriteLine("Carved: " + resultsNew[i]);
                            if (i != results.Length - 1)
                                file.WriteLine("");
                        }
                    }
                    else
                    {
                        for (int i = 0; i < results.Length; i++)
                        {
                            file.WriteLine("Target: " + targetName[i]);
                            file.WriteLine("Identified: " + results[i]);
                            if (i != results.Length - 1)
                                file.WriteLine("");
                        }
                    }

                file.WriteLine("-----------------------------\r\n");

                    if (unique.Count > 0)
                    {
                        file.WriteLine("File Index:");
                        //Entries in format = start + " \t\t " + finish + " \t\t " + Size of File + " \t\t " + file type;
                        file.WriteLine("---------------------------------------------------------------------------------------");
                        file.WriteLine("Start Index \t\t Finish Index \t\t Size of File \t\t File Type");
                        file.WriteLine("---------------------------------------------------------------------------------------");

                        //string[] foundLocs = foundResults.ToArray();

                        foreach (var item in unique.OrderBy(x => double.Parse(x.Substring(0, x.IndexOf(' ')))))
                            file.WriteLine(item);

                        if (shouldStop)
                        {
                            file.WriteLine("");
                            file.WriteLine("*** ANALYSIS ABORTED BY USER ***");
                            file.WriteLine("");
                        }

                        file.WriteLine("---------------------------------------------------------------------------------------");
                    }

                }

                System.Diagnostics.Process.Start(saveLocation + "LogFile.txt");

                if (!shouldStop)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        this.Close();
                    });
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not write Log File!\nError: " + ex, "Log File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        #endregion

        #endregion


        #region Processor Thread Functions

        private void CPUThread(int cpu, ref dataReader dataRead)
        {
            long count = 0;
            byte[] buffer = new byte[chunkSize];
            byte[] byteLocation = new byte[1];
            rewindCheck[cpu] = new int[1];
            bool rewindReq;
            bool firstPass;

            if (carveOp)
                byteLocation = new byte[chunkSize];

            double bytesRead;      // Location in file read
            while ((bytesRead = dataRead.GetChunk(buffer, ref count, ref totalProcessed)) > 0 && !shouldStop)   // Read into the buffer until end of file
            {
                chunkCount++;                                       // For each buffer used, increment count
                updateSegments();
                rewindReq = false;
                firstPass = true;
                rewindCheck[cpu][0] = 0;

                while (rewindReq == true || firstPass)
                {
                    if (firstPass == true)
                        firstPass = false;

                    if (carveOp)
                    {
                        updateGPUAct(cpu, true, false);
                        int[] tmpResults = Engine.CPUPFACAnalyse(carveOp, buffer, lookup, byteLocation, target.Length, longestTarget, fileLength, target.Length + 1, ref rewindCheck[cpu]);
                        updateGPUAct(cpu, false, false);

                        bool matchesFound = false;
                        for (int c = 0; c < results.Length; c++)
                        {
                            if (tmpResults[c*2] > 0)
                            {
                                matchesFound = true;
                                break;
                            }
                        }

                        if (matchesFound)
                        {
                            updateGPUAct(cpu, true, true);
                            smartCarve(cpu, 0, ref buffer, ref count, ref fileLength, ref target, ref lookup, ref targetEnd, ref lookupEnd, ref byteLocation, ref results);
                            updateGPUAct(cpu, false, false);
                            updateFound();
                        }
                    }
                    else
                    {
                        updateGPUAct(cpu, true, false);
                        int[] tmpResults = Engine.CPUPFACAnalyse(carveOp, buffer, lookup, new byte[1], target.Length, longestTarget, fileLength, target.Length + 1, ref rewindCheck[cpu]);
                        updateGPUAct(cpu, false, false);

                        for (int c = 0; c < tmpResults.Length; c++)
                            Interlocked.Add(ref results[c], tmpResults[c]);
                        updateFound();
                    }

                    Array.Clear(buffer, 0, buffer.Length);
                    Array.Clear(byteLocation, 0, byteLocation.Length);
                    rewindReq = false;
                }

                double Progress = (double)Math.Round(((totalProcessed / dataRead.GetFileSize()) * 100) / 10.0 * 10);
                updateProgress((int)Progress, totalProcessed, dataRead.GetFileSize());

            }

            buffer = new byte[1];
            updateGPUAct(cpu, true);
        }

        private void GPUThread(int gpu, int gpuCore, ref dataReader dataRead)
        {
            //MessageBox.Show(gpu.ToString() + " & " + gpuCore.ToString());
            long count = 0;
            byte[] buffer = new byte[chunkSize];
            byte[] byteLocation = new byte[1];
            int gpuID = gpu * gpuCoreCount + gpuCore;

            if (carveOp)
                byteLocation = new byte[chunkSize];

            double bytesRead;      // Location in file read
            while ((bytesRead = dataRead.GetChunk(buffer, ref count, ref totalProcessed)) > 0 && !shouldStop)   // Read into the buffer until end of file
            {
                chunkCount++;                                       // For each buffer used, increment count
                updateSegments();

                updateGPUAct(gpuID, true, false);
                GPUCollection[gpu].CopyToDevice(gpuCore, buffer);            // Copy buffer contents to GPU for processing  
                updateGPUAct(gpuID, false, false);

                if (carveOp)
                {
                    updateGPUAct(gpuID, true, false);
                    GPUCollection[gpu].LaunchPFACCarving(gpuCore);              //Carving Launch
                    updateGPUAct(gpuID, false, false);

                    bool matchesFound = false;
                    int[] carveResult = new int[results.Length];
                    for (int c = 0; c < carveResult.Length; c++)
                    {
                        Interlocked.Add(ref carveResult[c], GPUCollection[gpu].ReturnResultCount(gpuCore, c));
                        if (carveResult[c] > 0)
                        {
                            matchesFound = true;
                            byteLocation = GPUCollection[gpu].ReturnResult(gpuCore);
                            break;
                        }
                    }

                    if (matchesFound)
                    {
                        updateGPUAct(gpuID, true, true);
                        Parallel.For(0, procShare, i =>
                        {
                            smartCarve(gpu, i, ref buffer, ref count, ref fileLength, ref target, ref lookup, ref targetEnd, ref lookupEnd, ref byteLocation, ref results);
                        });
                        Task.WaitAll();
                        updateGPUAct(gpuID, false, true);
                    }
                }
                else
                {
                    updateGPUAct(gpuID, true, false);
                    GPUCollection[gpu].LaunchPFACCarving(gpuCore);              //Carving Launch
                    updateGPUAct(gpuID, false, false);

                    for (int c = 0; c < results.Length; c++)
                        Interlocked.Add(ref results[c], GPUCollection[gpu].ReturnResultCount(gpuCore, c));
                    updateFound();
                }

                Array.Clear(buffer, 0, buffer.Length);

                double Progress = (double)Math.Round(((totalProcessed / dataRead.GetFileSize()) * 100) / 10.0 * 10);
                updateProgress((int)Progress, totalProcessed, dataRead.GetFileSize());
            }
            buffer = new byte[1];
            byteLocation = new byte[1];

            updateGPUAct(gpuID, true);
        }

        #endregion


        #region Lookup Table Generation

        private void LookupTableGen(bool PFAC)
        {
            if (PFAC)
            {
                if (carveOp)
                {
                    target = new Byte[targetHeader.Count + targetFooter.Count][];
                    targetEnd = new Byte[targetFooter.Count][];

                    int i = 0;
                    for (int j = 0; j < targetHeader.Count; j++, i += 2)       // Translate Search Targets into Bytes
                    {
                        target[i] = Engine.GetBytes(targetHeader[j]);
                        targetEnd[j] = Engine.GetBytes(targetFooter[j]);
                        target[i + 1] = Engine.GetBytes(targetFooter[j]);
                    }
                }
                else
                {
                    target = new Byte[targetHeader.Count][];
                    for (int i = 0; i < target.Length; i++)            // Translate Search Targets into Bytes
                        target[i] = Engine.GetBytes(targetHeader[i]);
                }

                lookup = Engine.pfacLookupCreate(target);           // Create Lookup for Target
            }
            else
            {
                target = new Byte[targetHeader.Count][];   // Translate Search Targets into Bytes
                for (int i = 0; i < target.Length; i++)
                    target[i] = Engine.GetBytes(targetHeader[i]);
                lookup = Engine.bmLookupCreate(target);           // Create Lookup for Target

                if (carveOp)
                {
                    targetEnd = new Byte[targetFooter.Count][];    // Translate Search Target Footers into Bytes
                    for (int i = 0; i < targetEnd.Length; i++)
                        targetEnd[i] = Engine.GetBytes(targetFooter[i]);
                    lookupEnd = Engine.bmLookupCreate(targetEnd);     // Create Lookup for Target Footers
                }
            }
        }

        #endregion


        #region File Carving Operations

        private void smartCarve(int gpu, int procNo, ref byte[] buffer, ref long count, ref int fileLength, ref byte[][] target, ref int[][] lookup, ref byte[][] targetEnd, ref int[][] lookupEnd, ref byte[] resultLoc, ref int[] results)
        {
            int i = (procNo * (resultLoc.Length / procShare));
            int end = ((procNo + 1) * (resultLoc.Length / procShare));

            while (i < end && !shouldStop)
            {
                //MessageBox.Show("Processor " + procNo + "\nStart: " + i + " Finish: " + end);

                if ((int)resultLoc[i] != 0 && (int)resultLoc[i] % 2 != 0 && (int)resultLoc[i] < target.Length) // Experimental - last condition
                {
                    int headerType = (int)resultLoc[i];
                    int fileIndex = ((headerType + 1) / 2) - 1;
                    int footerType = 0;
                    for (int j = 0; j < targetName.Count; j++)
                    {
                        if (targetName[j] == targetName[fileIndex])
                        {
                            footerType = (j * 2) + 2;
                            break;
                        }
                    }

                    int searchRange = i + fileLength;
                    if (searchRange > buffer.Length)
                        searchRange = buffer.Length;

                    bool nextHeaderFound = false;
                    int nextHeader = 0;
                    int fileEnd = 0;

                    for (int j = i + 1; j < searchRange; j++)
                    {
                        if (resultLoc[j] == headerType && !nextHeaderFound)
                        {
                            nextHeader = j - 1;
                            nextHeaderFound = true;
                        }

                        if (targetName[fileIndex] == "sqldb")
                        {
                            fileEnd = i + sqldbSearchLength(ref buffer, i);
                            if (fileEnd > buffer.Length)
                                fileEnd = 0;
                            if (buffer[fileEnd] != 0x38 && buffer[fileEnd + 1] != 0x38 && buffer[fileEnd + 1] != 0x3B)
                            {
                                CarveFile(buffer, count, fileIndex, i, fileEnd, "");
                                break;
                            }
                            else
                                fileEnd = 0;
                        }
                        else
                        {
                            if ((int)resultLoc[j] == footerType)
                            {
                                fileEnd = j + targetEnd[fileIndex].Length;
                                fileEnd = footerAdjust(fileEnd, targetName[fileIndex]);
                                if (fileEnd > buffer.Length)
                                    fileEnd = 0;
                                if (buffer[fileEnd] != 0x38 && buffer[fileEnd + 1] != 0x38 && buffer[fileEnd + 1] != 0x3B)
                                {
                                    CarveFile(buffer, count, fileIndex, i, fileEnd, "");
                                    break;
                                }
                                else
                                    fileEnd = 0;
                            }
                        }
                    }

                    if (fileEnd == 0 && nextHeaderFound)
                        CarveFile(buffer, count, fileIndex, i, nextHeader, "fragmented");
                    else if (fileEnd == 0 && !nextHeaderFound && searchRange != buffer.Length)
                        CarveFile(buffer, count, fileIndex, i, searchRange, "partial");

                }

                i++;
            }
        }

        private void CarveFile(byte[] buffer, double count, int fileIndex, int start, int finish, string tag)
        {
            string filePath = saveLocation + targetName[fileIndex] + "/";
            if (!Directory.Exists(filePath))
                Directory.CreateDirectory(filePath);
            if (!File.Exists(filePath + (count + start).ToString() + "." + targetName[fileIndex]))
            {
                try
                {
                    byte[] fileData = new byte[finish - start];
                    Array.Copy(buffer, start, fileData, 0, finish - start);

                    float fileSize = (finish - start);
                    string sizeFormat = "bytes";
                    if (fileSize > 1024)
                    {
                        fileSize = fileSize / 1024;
                        sizeFormat = "KB";
                    }
                    if (fileSize > 1024)
                    {
                        fileSize = fileSize / 1024;
                        sizeFormat = "MB";
                    }
                    if (fileSize > 1024)
                    {
                        fileSize = fileSize / 1024;
                        sizeFormat = "GB";
                    }
                    String newEntry = (count + start).ToString() + " \t\t " + (count + finish).ToString() + " \t\t " + Math.Round(fileSize, 4).ToString() + " " + sizeFormat + " \t\t " + tag + " " + targetName[fileIndex];
                    foundResults.Add(newEntry);

                    File.WriteAllBytes(filePath + (count + start).ToString() + tag + "." + targetName[fileIndex], fileData);
                    Interlocked.Increment(ref results[fileIndex]);
                    updateFound();
                }
                catch { }
            }
        }

        private int sqldbSearchLength(ref byte[] buffer, int count)
        {
            // Page Size (int16, offset 16) 
            byte[] dbPageLength = new byte[2];
            Array.Copy(buffer, count+16, dbPageLength, 0, 2);
            Array.Reverse(dbPageLength);
            int pageLength = BitConverter.ToInt16(dbPageLength, 0);
            if (pageLength == 1)
                pageLength = 65536;

            // Page Count (int32, offset 28, supported after 3.7)
            byte[] dbPageCount = new byte[4];
            Array.Copy(buffer, count + 28, dbPageCount, 0, 4);
            Array.Reverse(dbPageCount);
            int pageCount = BitConverter.ToInt32(dbPageCount, 0);

            // Total Freelist Pages (int32, offset 36)
            if(pageCount==0)
            {
                byte[] dbFreelistPageCount = new byte[4];
                Array.Copy(buffer, count + 36, dbFreelistPageCount, 0, 4);
                Array.Reverse(dbFreelistPageCount);
                pageCount = BitConverter.ToInt32(dbFreelistPageCount, 0);
            }

            int dbSize = pageCount * pageLength;

            return dbSize;
        }

        private void genericCarve(int gpu, ref byte[] buffer, ref long count, ref int fileLength, ref byte[][] target, ref int[][] lookup, ref byte[][] targetEnd, ref int[][] lookupEnd, byte[] resultLoc, ref int[] results)
        {
            //Interlocked.Increment(ref activeThreads[gpu]);

            int i = 0;

            while (i < resultLoc.Length && !shouldStop)
            {
                if ((int)resultLoc[i] != 0 && (int)resultLoc[i] % 2 != 0)
                {
                    int headerType = (int)resultLoc[i];
                    int fileIndex = ((headerType + 1) / 2) - 1;
                    int footerType = 0;
                    for (int j = 0; j < targetName.Count; j++)
                    {
                        if (targetName[j] == targetName[fileIndex])
                        {
                            footerType = (j*2)+2;
                            break;
                        }
                    }

                    int searchRange = i + fileLength;
                    if (searchRange > buffer.Length)
                        searchRange = buffer.Length;

                    for (int j = i; j < searchRange; j++)
                    {
                        if ((int)resultLoc[j] == footerType)
                        {
                            int fileEnd = j + targetEnd[fileIndex].Length;
                            fileEnd = footerAdjust(fileEnd, targetName[fileIndex]);

                            if (buffer[fileEnd] != 0x38 && buffer[fileEnd + 1] != 0x38 && buffer[fileEnd + 1] != 0x3B)
                            {
                                string filePath = saveLocation + targetName[fileIndex] + "/";
                                byte[] fileData = new byte[fileEnd - i];
                                Array.Copy(buffer, i, fileData, 0, fileEnd - i);
                                if (!Directory.Exists(filePath))
                                    Directory.CreateDirectory(filePath);
                                if (!File.Exists(filePath + (count + i).ToString() + "." + targetName[fileIndex]))
                                {
                                    try
                                    {
                                        float fileSize = (fileEnd - i);
                                        string sizeFormat = "bytes";
                                        if (fileSize > 1024)
                                        {
                                            fileSize = fileSize / 1024;
                                            sizeFormat = "KB";
                                        }
                                        if (fileSize > 1024)
                                        {
                                            fileSize = fileSize / 1024;
                                            sizeFormat = "MB";
                                        }
                                        if (fileSize > 1024)
                                        {
                                            fileSize = fileSize / 1024;
                                            sizeFormat = "GB";
                                        }
                                        String newEntry = (count + i).ToString() + " \t\t " + Math.Round(fileSize, 4).ToString() + " " + sizeFormat + " \t\t " + targetName[fileIndex];
                                        foundResults.Add(newEntry);

                                        File.WriteAllBytes(filePath + (count + i).ToString() + "." + targetName[fileIndex], fileData);
                                        results[fileIndex] += 1;
                                        updateFound();
                                    }
                                    catch { }
                                }
                                //i++;
                                break;
                            }
                        }
                    }
                        
                }

                i++;
            }
                
            //Interlocked.Decrement(ref activeThreads[gpu]);

            //if (activeThreads[gpu] == 0)
            //    carveDone[gpu].Set();
        }

        private int footerAdjust(int footer, string fileType)
        {
            int fileEnd = footer;

            if (fileType == "zip")
                fileEnd += 20;
            else if (fileType == "docx")
                fileEnd += 18;

            return fileEnd;
        }

        #endregion


        private void Carve_FormClosing(object sender, FormClosingEventArgs e)
        {
            if (e.CloseReason == CloseReason.UserClosing)
            {
                shouldStop = true;
                GC.Collect();
                this.Owner.Show();
            }
        }

    }
}
