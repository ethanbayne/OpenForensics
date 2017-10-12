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
using System.Xml;
using System.Xml.Serialization;

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

        public struct resultRecord
        {
            public double start, end;
            public float size;
            public string sizeformat, tag, filetype;

            public resultRecord(double start, double end, float size, string sizeformat, string tag,  string filetype)
            {
                this.start = start;
                this.end = end;
                this.size = size;
                this.sizeformat = sizeformat;
                this.tag = tag;
                this.filetype = filetype;
            }

            public resultRecord(double start, string tag, string filetype)
            {
                this.start = start;
                this.end = 0;
                this.size = 0;
                this.sizeformat = null;
                this.tag = tag;
                this.filetype = filetype;
            }

            public string printRecord()
            {
                if (end == 0)
                    return start.ToString() + " \t\t " + tag + " " + filetype;
                else
                    return start.ToString() + " \t\t " + end.ToString() + " \t\t " + Math.Round(size, 4).ToString() + " " + sizeformat + " \t\t " + tag + " " + filetype;
            }
        }


        #region Data Reading

        //dataReader - class that handles operations for reading from storage devices.
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

            public dataReader(string path, int peekLength)
            {
                if (path.StartsWith("\\\\.\\")) //If the path refers to a physical drive..
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
                    peek = (int)((peekLength / 512) * 512);
                    if (peek == 0)
                        peek = 512;
                }
                else  // Else the path specified is a file
                {
                    DDStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.None, readBuffer, FileOptions.SequentialScan);
                    fileSize = (ulong)DDStream.Length;

                    peek = peekLength;
                }

                sectionEnd = new byte[peek];
            }

            public int GetChunk(byte[] buffer, ref long count, ref double totalProcessed)
            {
                //Lock the process so that only one process can read from the drive at any given point.
                lock (locker)
                {
                    count = DDStream.Position;  //Store count as the current byte position in the file/drive
                    int toRead = buffer.Length;

                    // Physical Drive Read Boundary Tests
                    if (physicalDrive && (ulong)count != fileSize && count > (long)(fileSize - 512))    // If data read position on physical drive is less than the 512 byte minimum
                    {
                        DDStream.Flush();
                        DDStream.Position = (long)(fileSize - 512); // Rewind 512 bytes from the end to ensure data can be analysed
                        count = DDStream.Position;
                        toRead = 512;
                    }
                    if ((ulong)(count + toRead) > fileSize) // Ensure the amount being read does not exceed file/drive size
                        toRead = (int)(fileSize - (ulong)count);

                    int result = 0;
                    if (toRead != 0)    // If there's data to read..
                    {
                        int buffSplit = 0;
                        int oddBytes = 0;
                        bool includingPeek = true;

                        // If it's not the final sector and bytes to read is greater than the peek window, 
                        // Copy the window buffer from the last section to the beginning and 
                        // append the length to read by the peek length.
                        if ((ulong)(count + toRead) != fileSize && count > 0 && toRead > peek)
                        {
                            toRead -= peek;
                            count -= peek;
                            Array.Copy(sectionEnd, buffer, peek);
                        }
                        else
                            includingPeek = false;

                        if (toRead <= readLength) // If the data to read is less than the size of one read instruction, only use 1 read instruction
                        {
                            buffSplit = 1;
                            oddBytes = toRead;
                        }
                        else // Else, divide the data to read to even read instructions
                        {
                            buffSplit = toRead / readLength;
                            oddBytes = toRead % readLength; // Account for odd bytes that may still remain.
                            if (oddBytes != 0)
                                buffSplit++;
                        }

                        long currentPos = count;

                        // Create a read queue to read data from the storage device.
                        ConcurrentQueue<int> queue = new ConcurrentQueue<int>();
                        for (int n = 0; n < buffSplit; n++)
                        {
                            int x = n;
                            int workLoad = readLength;
                            if (x == buffSplit - 1 && oddBytes != 0)
                                workLoad = oddBytes;
                            if (includingPeek)
                                queue.Enqueue(Process(DDStream, buffer, peek + (x * readLength), workLoad));
                            else
                                queue.Enqueue(Process(DDStream, buffer, x * readLength, workLoad));
                        }
                        
                        // Create new scheduler to read data. Use readQueues to manage levels of concurrency.
                        LimitedConcurrencyLevelTaskScheduler scheduler = new LimitedConcurrencyLevelTaskScheduler(readQueues);
                        TaskFactory factory = new TaskFactory(scheduler);

                        // Start actioning the read queue
                        while (!queue.IsEmpty)
                        {
                            factory.StartNew(() =>
                            {
                                int task;
                                if (queue.TryDequeue(out task))
                                    result += task;

                            }, TaskCreationOptions.PreferFairness);
                        }

                        // Cache end of segment in memory to preserve DDStream cache
                        if (DDStream.Position != (long)fileSize)
                            Array.Copy(buffer, buffer.Length - peek, sectionEnd, 0, peek);
                    }

                    // Update the total progressed as being the current position in data.
                    totalProcessed = DDStream.Position;
                    return result;
                }
            }

            private static int Process(FileStream fs, byte[] array, int start, int length)
            {
                //Thread.CurrentThread.Priority = ThreadPriority.AboveNormal;
                return fs.Read(array, start, length);
            }

            // Past chunk is a legacy function that is able to retrieve a chunk of past data in the file stream and then
            // return the position to the last known position.
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
            
            // Function for carving file from source device.
            public byte[] RetrieveFile(long startLocation, long endLocation)
            {
                lock (locker)
                {
                    DDStream.Flush();
                    DDStream.Position = startLocation;


                    int fileSize = (int)(endLocation - startLocation);
                    byte[] file = new byte[fileSize];
                    int buffSplit = 0;
                    int oddBytes = 0;
                    int result = 0;

                    if (fileSize <= readLength) // If the data to read is less than the size of one read instruction, only use 1 read instruction
                    {
                        buffSplit = 1;
                        oddBytes = fileSize;
                    }
                    else // Else, divide the data to read to even read instructions
                    {
                        buffSplit = fileSize / readLength;
                        oddBytes = fileSize % readLength; // Account for odd bytes that may still remain.
                        if (oddBytes != 0)
                            buffSplit++;
                    }

                    // Create a read queue to read data from the storage device.
                    ConcurrentQueue<int> queue = new ConcurrentQueue<int>();
                    for (int n = 0; n < buffSplit; n++)
                    {
                        int x = n;
                        int workLoad = readLength;
                        if (x == buffSplit - 1 && oddBytes != 0)
                            workLoad = oddBytes;
                        queue.Enqueue(Process(DDStream, file, x * readLength, workLoad));
                    }

                    // Create new scheduler to read data. Use readQueues to manage levels of concurrency.
                    LimitedConcurrencyLevelTaskScheduler scheduler = new LimitedConcurrencyLevelTaskScheduler(readQueues);
                    TaskFactory factory = new TaskFactory(scheduler);

                    // Start actioning the read queue
                    while (!queue.IsEmpty)
                    {
                        factory.StartNew(() =>
                        {
                            int task;
                            if (queue.TryDequeue(out task))
                                result += task;

                        }, TaskCreationOptions.PreferFairness);
                    }

                    return file;
                }
            }

            // Safely closes the file stream.
            public void CloseFile()
            {
                DDStream.Close();
            }

            // Retrieve file size
            public ulong GetFileSize()
            {
                return (ulong)fileSize;
            }
        }

        #endregion


        #region Form Load and Special Functions

        public Analysis()
        {
            InitializeComponent();
        }

        public class Input
        {
            public string TestType { get; set; }
            public string GPGPU { get; set; }
            public List<int> gpus { get; set; }
            public long maxGPUMem { get; set; }
            public string FilePath { get; set; }
            public string CaseName { get; set; }
            public string EvidenceName { get; set; }
            public string saveLocation { get; set; }
            public string CarveFilePath { get; set; }
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
        private string CarveFilePath;
        private List<string> targetName;
        private List<string> targetHeader;
        private List<string> targetFooter;

        private Label[] gpuLabel;

        // Hard coded chunk size and expected maximum file length.
        // TO-DO: State fileLength in config file.
        private uint chunkSize = 100 * 1048576;
        private int fileLength = 20 * 1048576;
        private uint resultCache = 1048576;

        private Byte[][] target;
        private int[][] lookup;
        private Byte[][] targetEnd;

        private int longestTarget;

        private Stopwatch watch;
        private int[] results;
        private double totalProcessed;
        private int carveProcessed;
        private uint chunkCount;
        private ConcurrentBag<resultRecord> foundResults = new ConcurrentBag<resultRecord>();
        private List<resultRecord> carvableFiles = new List<resultRecord>();

        private List<Engine> GPUCollection = new List<Engine>();

        public Input InputSet
        {
            set
            {
                TestType = value.TestType;
                GPGPU = value.GPGPU;
                gpus = value.gpus;
                maxGPUMem = value.maxGPUMem;
                FilePath = value.FilePath;
                CaseName = value.CaseName;
                EvidenceName = value.EvidenceName;
                saveLocation = value.saveLocation;
                CarveFilePath = value.CarveFilePath;
                targetName = value.targetName;
                targetHeader = value.targetHeader;
                targetFooter = value.targetFooter;
            }
        }

        private void Carve_Load(object sender, EventArgs e)
        {
            try
            {

                // Interface updates for tech
                string techUsed = "";
                if (TestType == "CPU")
                    techUsed = " (" + lpCount + " Logical Cores)";
                else
                {
                    int maxGPUThread = (int)((maxGPUMem * 0.8) / (chunkSize + ((targetHeader.Count * 2) * resultCache)));
                    gpuCoreCount = Math.Min(lpCount / gpus.Count, maxGPUThread);
                    //gpuCoreCount = 1;   // Force GPU concurrency to a certain value.

                    if (gpus.Count > 1)
                        techUsed = " (" + GPGPU + " - " + gpus.Count + " GPUs - running " + gpuCoreCount + " threads each)";
                    else
                        techUsed = " (" + GPGPU + " - running " + gpuCoreCount + " threads)";
                }

                // Initial setup of GPU status
                DrawGPUStatus();

                if (CarveFilePath == "")
                {
                    // Records analysis information
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

                    // Start the main analysis processing thread
                    Thread analysis = new Thread(new ThreadStart(FileAnalysis));
                    analysis.Start();
                }
                else
                {
                    Thread carve = new Thread(new ThreadStart(CarveOnly));
                    carve.Start();
                }
            }
            catch (Exception ex) // Pokémon error catching, gotta catch them all!
            {
                MessageBox.Show(ex.ToString(), "Carving Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        public void CarveClose()
        {
            if (!shouldStop)
            {
                this.Invoke((MethodInvoker)delegate
                {
                    this.Close();
                });
            }
        }

        // Main timer function for recording time required for searching
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

        // Populates the main interface with status boxes that represent each active processing thread.
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


        // Update header text that outlines test parameters
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

        // Updates the amount of segments analysed.
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

        // Updates the amount of files carved or identified.
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

        // Update the visual representation of GPU status.
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
                                this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightBlue;  // Carving files and saving to target drive.
                            else
                                this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightGreen; // Actively Searching.
                        else
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.Green;  // Idle.
                    });
                }
                else
                {
                    if (status)
                        if (carving)
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightBlue;  // Carving files and saving to target drive.
                        else
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.LightGreen; // Actively Searching.
                    else
                        this.gpuLabel[gpu].BackColor = System.Drawing.Color.Green;  // Idle.
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
                            this.gpuLabel[gpu].BackColor = System.Drawing.Color.DimGray;    // Processing thread finished.
                        });
                    }
                    else
                    {
                        this.gpuLabel[gpu].BackColor = System.Drawing.Color.DimGray;    // Processing thread finished.
                    }
                }
            }
            catch (Exception)
            { }
        }

        // Update percentage and time values on interface
        private void updateProgress(int percent, double position, double total)
        {
            try
            {
                if (CarveFilePath == "")
                {
                    double timeElapsed = (int)Math.Round(watch.Elapsed.TotalSeconds, 2);
                    TimeSpan timeElapsedSpan = new TimeSpan(0, 0, (int)Math.Round(watch.Elapsed.TotalSeconds, 2));
                    TimeSpan timeRemainingSpan = new TimeSpan(0, 0, (int)((timeElapsed / percent) * (100 - percent)));

                    string formattedElapsed = string.Format("{0:00}:{1:00}:{2:00}", timeElapsedSpan.Hours, timeElapsedSpan.Minutes, timeElapsedSpan.Seconds);
                    string formattedRemaining = "--:--:--";
                    if (percent > 0)
                        formattedRemaining = string.Format("{0:00}:{1:00}:{2:00}", timeRemainingSpan.Hours, timeRemainingSpan.Minutes, timeRemainingSpan.Seconds);

                    if (this.pbProgress.InvokeRequired)
                    {
                        this.Invoke((MethodInvoker)delegate
                        {
                            this.lblTimeElapsedValue.Text = formattedElapsed;
                            this.lblTimeRemainingValue.Text = formattedRemaining;
                        });
                    }
                    else
                    {
                        this.lblTimeElapsedValue.Text = formattedElapsed;
                        this.lblTimeRemainingValue.Text = formattedRemaining;
                    }
                }

                if (this.pbProgress.InvokeRequired)
                {
                    this.Invoke((MethodInvoker)delegate
                    {
                        this.pbProgress.Value = percent;
                        this.lblProgress.Text = percent + "%";
                        this.lblProcess.Text = "Processing: " + position + " / " + total;
                    });
                }
                else
                {
                    this.pbProgress.Value = percent;
                    this.lblProgress.Text = percent + "%";
                    this.lblProcess.Text = "Processing: " + position + " / " + total;
                }              
            }
            catch (Exception)
            { }
        }

        #endregion


        #region Analysis Command

        // Main analysis function
        private void FileAnalysis()
        {

            if (TestType == "CPU")
            {
                #region DD Carve - CPU Section

                updateHeader("Analysis Started using " + TestType + " (Cores: " + lpCount + ")");

                // Uncomment for single-threaded CPU operation
                //lpCount = 1;

                String path = FilePath;                                 // Target DD File

                LookupTableGen();

                results = new int[target.Length/2];           // Number of Results Found

                // Figure out what will be the longest header
                for (int i = 0; i < target.Length; i++)
                    if (target[i] != null && target[i].Length > longestTarget)
                        longestTarget = target[i].Length;

                // Each logical CPU core used will rely on the same thread to carve
                procShare = 1;

                // Set up data reader
                dataReader dataRead = new dataReader(FilePath, fileLength);

                ulong fileSize = dataRead.GetFileSize();

                // Start stopwatch, open file defined by user
                double time = MeasureTime(() =>
                {
                    // Launch a thread for each logical core of the CPU
                    Parallel.For(0, lpCount, async i =>
                    {
                        await CPUThread(i, dataRead);
                    });
                    Task.WaitAll();
                });

                // Prepare the results file
                PrepareResults(fileSize, time);

                if (carvableFiles.Count > 0)
                    if (MessageBox.Show("Reproduce found files? (" + carvableFiles.Count + " carvable files found)", "File Carving", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                        carveResults(dataRead);

                // When all threads have finished, close file
                dataRead.CloseFile();

                CarveClose();

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

                updateHeader("Analysis Started using " + gpuText);

                LookupTableGen();
                
                results = new int[target.Length / 2];                  // Number of Results Found

                totalProcessed = 0;
                chunkCount = 0;                                 // Number of segments processed

                // Find longest header
                for (int i = 0; i < target.Length; i++)
                    if (target[i] != null && target[i].Length > longestTarget)
                        longestTarget = target[i].Length;

                // Create a GPU object for each GPU selected
                foreach(int GPUid in gpus)
                {
                    Engine gpu = new Engine(GPUid, gpuCoreCount, target, lookup, longestTarget, fileLength, chunkSize);
                    GPUCollection.Add(gpu);
                }

                // Share logical CPU cores between GPUs employed.
                procShare = Math.Min(Math.Max(lpCount / gpuCoreCount / GPUCollection.Count, 1), lpCount);
                //procShare = Math.Min(Math.Max(lpCount / GPUCollection.Count, 1), lpCount);  // Divide between employed GPUs

                //procShare = 1; // Force logical CPU cores per GPU

                // Set up data reader
                dataReader dataRead = new dataReader(FilePath, fileLength);

                ulong fileSize = dataRead.GetFileSize();

                // Start stopwatch, open file defined by user
                double time = MeasureTime(() =>
                {
                    // For each GPU employed, launch a dedicated thread
                    Parallel.For(0, GPUCollection.Count, i =>
                    {
                        Parallel.For(0, gpuCoreCount, async j =>
                        {
                            await GPUThread(i, j, dataRead);
                        });
                    });
                    Task.WaitAll();
                });

                // When all threads complete, free memory and close file
                for (int i = 0; i < GPUCollection.Count; i++)
                {
                    GPUCollection[i].FreeAll();                                   // Free all GPU resources
                    GPUCollection[i].HostFreeAll();
                }

                // Prepare the results file
                PrepareResults(fileSize, time);

                if (carvableFiles.Count > 0)
                    if (MessageBox.Show("Reproduce found files? (" + carvableFiles.Count + " carvable files found)", "File Carving", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                        carveResults(dataRead);

                dataRead.CloseFile();

                CarveClose();
                #endregion
            }
        }

        private void CarveOnly()
        {
            carvableFiles = loadCarvableLocations<List<resultRecord>>(saveLocation + CarveFilePath);
            dataReader dataRead = new dataReader(FilePath, fileLength);
            carveResults(dataRead);
            dataRead.CloseFile();
            CarveClose();
        }

        #region Result Prep

        // Prepares the results file
        private void PrepareResults(ulong fileSize, double time)
        {
            updateHeader("Preparing results...");

            try
            {
                using (System.IO.StreamWriter file = new System.IO.StreamWriter(saveLocation + "LogFile.txt", true))
                {
                    int duplicates = 0;
                    int[] resultsNew = new int[results.Length];
                    List<resultRecord> tmpFoundLocs = new List<resultRecord>();
                    List<resultRecord> foundLocs = new List<resultRecord>();
                    tmpFoundLocs.AddRange(foundResults.ToArray());
                    tmpFoundLocs.Sort((s1, s2) => s1.start.CompareTo(s2.start));
                    for (int i = 0; i < tmpFoundLocs.Count;)
                    {
                        if (i < (tmpFoundLocs.Count - 1) && tmpFoundLocs[i].start == tmpFoundLocs[i + 1].start)
                        {
                            if (tmpFoundLocs[i].tag.Contains("partial") && tmpFoundLocs[i].tag.Contains("fragmented"))
                                foundLocs.Add(tmpFoundLocs[i + 1]);
                            else if (tmpFoundLocs[i + 1].tag.Contains("partial") && tmpFoundLocs[i + 1].tag.Contains("fragmented"))
                                foundLocs.Add(tmpFoundLocs[i]);
                            else if (tmpFoundLocs[i].end > tmpFoundLocs[i + 1].end)
                                foundLocs.Add(tmpFoundLocs[i]);
                            else
                                foundLocs.Add(tmpFoundLocs[i + 1]);

                            duplicates++;
                            i += 2;
                        }
                        else
                        {
                            foundLocs.Add(tmpFoundLocs[i]);
                            if (tmpFoundLocs[i].end != 0)
                                carvableFiles.Add(tmpFoundLocs[i]);
                            i++;
                        }

                        int nameIndex = targetName.IndexOf(foundLocs[foundLocs.Count - 1].filetype.ToString());
                        resultsNew[nameIndex]++;
                    }
                    tmpFoundLocs.Clear();

                    file.WriteLine("Statistics:");
                    file.WriteLine("-----------------------------");
                    file.WriteLine("Bytes Analysed: " + fileSize + " (" + Math.Round((double)fileSize / 1000000000, 2).ToString() + " GB)");
                    file.WriteLine("File Segments : " + chunkCount);
                    file.WriteLine("");
                    if (shouldStop)
                    {
                        file.WriteLine("ANALYSIS ABORTED EARLY BY USER - RESULTS INCOMPLETE");
                        file.WriteLine("");
                    }
                    int total = 0;


                    foreach (int count in resultsNew)
                        total += count;
                    file.WriteLine("Total Files Found: " + total);

                    string timeInfo = "";
                    if (Math.Round(time, 2) >= 60)
                    {
                        TimeSpan timeElapsedSpan = new TimeSpan(0, 0, (int)Math.Round(watch.Elapsed.TotalSeconds, 2));
                        timeInfo = string.Format(" ({0:00}:{1:00}:{2:00})", timeElapsedSpan.Hours, timeElapsedSpan.Minutes, timeElapsedSpan.Seconds);
                    }

                    file.WriteLine("Processing Time: " + Math.Round(time, 2).ToString() + " seconds" + timeInfo);
                    file.WriteLine("Processing Rate: " + Math.Round(((chunkCount * (chunkSize / 1048576)) / time), 2).ToString() + " MB/sec");
                    file.WriteLine("-----------------------------\r\n");
                    file.WriteLine("");
                    file.WriteLine("Result Breakdown:");
                    file.WriteLine("-----------------------------");

                    for (int i = 0; i < resultsNew.Length; i++)
                    {
                        file.WriteLine("Target: " + targetName[i]);
                        file.WriteLine("Carved: " + resultsNew[i]);
                        if (i != results.Length - 1)
                            file.WriteLine("");
                    }

                    file.WriteLine("-----------------------------\r\n");

                    if (foundLocs.Count > 0)
                    {
                        file.WriteLine("File Index:");
                        //Entries in format = start + " \t\t " + finish + " \t\t " + Size of File + " \t\t " + file type;
                        file.WriteLine("---------------------------------------------------------------------------------------");
                        file.WriteLine("Start Index \t\t Finish Index \t\t Size of File \t\t File Type");
                        file.WriteLine("---------------------------------------------------------------------------------------");

                        //string[] foundLocs = foundResults.ToArray();

                        foreach (resultRecord item in foundLocs)
                            file.WriteLine(item.printRecord());

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

            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not write Log File!\nError: " + ex, "Log File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            saveCarvableLocations(carvableFiles, saveLocation + "CarvableFileData.of");
        }

        #endregion

        #endregion


        #region Processor Thread Functions

        private Task<Boolean> CPUThread(int cpu, dataReader dataRead)
        {
            long count = 0;
            byte[] buffer = new byte[chunkSize];
            byte[] foundID = new byte[resultCache];
            int[] foundLoc = new int[resultCache];

            double bytesRead;      // Location in file read
            while ((bytesRead = dataRead.GetChunk(buffer, ref count, ref totalProcessed)) > 0 && !shouldStop)   // Request data chunk until end of file
            {
                chunkCount++;   // Add one to chunk count
                updateSegments();   // UI update

                // Launch analysis
                updateGPUAct(cpu, true, false);
                int[] tmpResults = Engine.CPUPFACAnalyse(buffer, lookup, ref foundID, ref foundLoc, target.Length);
                updateGPUAct(cpu, false, false);

                // Are there any matches found?
                bool matchesFound = false;
                for (int c = 0; c < results.Length; c++)
                {
                    if (tmpResults[c] > 0)
                    {
                        matchesFound = true;
                        break;
                    }
                }

                // Carve if mactches exist, update UI
                if (matchesFound)
                {
                    updateGPUAct(cpu, true, true);
                    int resultCount = 0;
                    for (int z = 0; z < foundID.Length; z++)
                        if (foundID[z] == 0)
                        {
                            Array.Resize(ref foundID, z);
                            Array.Resize(ref foundLoc, z);
                            resultCount = z;
                        }
                    Array.Sort(foundLoc, foundID);
                    ProcessLocations(cpu, 0, ref buffer, ref count, ref foundID, ref foundLoc);
                    updateGPUAct(cpu, false, false);
                    updateFound();
                }

                // Clear buffer and byteLocation for reuse
                Array.Clear(buffer, 0, buffer.Length);
                //Array.Clear(foundID, 0, foundID.Length);   //< Attempts to resize array? unusual.. GC?
                //Array.Clear(foundLoc, 0, foundLoc.Length);
                foundID = new byte[resultCache];
                foundLoc = new int[resultCache];

                // Update progress
                double Progress = (double)Math.Round(((totalProcessed / dataRead.GetFileSize()) * 100) / 10.0 * 10);
                updateProgress((int)Progress, totalProcessed, dataRead.GetFileSize());
            }

            // After thread is done, minimise buffer and byteLocation to 1, update UI
            buffer = new byte[1];
            foundID = new byte[1];
            foundLoc = new int[1];
            updateGPUAct(cpu, true);

            return Task.FromResult(true);
        }

        private Task<Boolean> GPUThread(int gpu, int gpuCore, dataReader dataRead)
        {
            //MessageBox.Show(gpu.ToString() + " & " + gpuCore.ToString());
            long count = 0;
            byte[] buffer = new byte[chunkSize];
            byte[] foundID = new byte[resultCache];
            int[] foundLoc = new int[resultCache];
            int gpuID = gpu * gpuCoreCount + gpuCore;

            double bytesRead;      // Location in file read
            while ((bytesRead = dataRead.GetChunk(buffer, ref count, ref totalProcessed)) > 0 && !shouldStop)   // Read into the buffer until end of file
            {
                chunkCount++;                                       // For each buffer used, increment count
                updateSegments();

                updateGPUAct(gpuID, true, false);
                GPUCollection[gpu].CopyToDevice(gpuCore, buffer);            // Copy buffer contents to GPU for processing  
                updateGPUAct(gpuID, false, false);

                // Launch file carving on GPU
                updateGPUAct(gpuID, true, false);
                GPUCollection[gpu].LaunchPFACCarving(gpuCore);
                updateGPUAct(gpuID, false, false);

                // Validate whether matches were found
                bool matchesFound = false;
                int[] carveResult = new int[results.Length];
                for (int c = 0; c < carveResult.Length; c++)
                {
                    Interlocked.Add(ref carveResult[c], GPUCollection[gpu].ReturnResultCount(gpuCore, c));
                    if (carveResult[c] > 0)
                    {
                        matchesFound = true;
                        //byteLocation = GPUCollection[gpu].ReturnResult(gpuCore);
                        foundID = GPUCollection[gpu].ReturnResultID(gpuCore);
                        foundLoc = GPUCollection[gpu].ReturnResultLoc(gpuCore);
                        break;
                    }
                }

                // If matches found, perform file carving on CPU
                if (matchesFound)
                {
                    updateGPUAct(gpuID, true, true);
                    int resultCount = 0;
                    for (int z = 0; z < foundID.Length; z++)
                        if (foundID[z] == 0)
                        {
                            Array.Resize(ref foundID, z);
                            Array.Resize(ref foundLoc, z);
                            resultCount = z;
                        }
                    Array.Sort(foundLoc, foundID);
                    Parallel.For(0, procShare, i =>
                    {
                        ProcessLocations(gpu, i, ref buffer, ref count, ref foundID, ref foundLoc);
                    });
                    Task.WaitAll();
                    updateGPUAct(gpuID, false, true);
                }

                // Clear buffer
                Array.Clear(buffer, 0, buffer.Length);

                // Update progress
                double Progress = (double)Math.Round(((totalProcessed / dataRead.GetFileSize()) * 100) / 10.0 * 10);
                updateProgress((int)Progress, totalProcessed, dataRead.GetFileSize());
            }

            // After thread is done, minimise buffer and byteLocation to 1, update UI
            buffer = new byte[1];
            foundID = new byte[1];
            foundLoc = new int[1];
            updateGPUAct(gpuID, true);

            return Task.FromResult(true);
        }

        #endregion


        #region Lookup Table Generation

        // Lookup Table generation for PFAC and BM
        private void LookupTableGen()
        {
            target = new Byte[targetHeader.Count + targetFooter.Count][];
            targetEnd = new Byte[targetFooter.Count][];

            int i = 0;
            for (int j = 0; j < targetHeader.Count; j++, i += 2)       // Translate Search Targets into Bytes
            {
                target[i] = Engine.GetBytes(targetHeader[j]);
                if (targetFooter[j] != null)
                {
                    targetEnd[j] = Engine.GetBytes(targetFooter[j]);
                    target[i + 1] = Engine.GetBytes(targetFooter[j]);
                }
            }

            lookup = Engine.pfacLookupCreate(target);           // Create Lookup for Target
        }

        #endregion


        #region File Carving Operations

        // Result processing. Buffer is divided between logical cores assigned to file carve.
        private void ProcessLocations(int gpu, int numThreads, ref byte[] buffer, ref long count, ref byte[] resultID, ref int[] resultLoc)
        {
            int i = (numThreads * (resultLoc.Length / procShare));
            int end = ((numThreads + 1) * (resultLoc.Length / procShare));

            while (i < end && !shouldStop)
            {
                if (resultID[i] % 2 != 0)
                {
                    int headerType = (int)resultID[i];
                    int fileIndex = ((headerType + 1) / 2) - 1;
                    int footerType = 0;
                    for (int y = 0; y < targetName.Count; y++)
                    {
                        if (targetName[y] == targetName[fileIndex])
                        {
                            footerType = (y * 2) + 2;
                            break;
                        }
                    }

                    if (targetEnd[fileIndex] != null)
                    {

                        int searchRange = resultID.Length;

                        bool nextHeaderFound = false;
                        int nextHeader = 0;
                        int fileEnd = 0;

                        for (int j = (i + 1); j < searchRange; j++)
                        {
                            if (resultID[j] == headerType)
                            {
                                nextHeader = resultLoc[j] - 1;
                                nextHeaderFound = true;
                                break;
                            }
                            else if (resultID[j] == footerType)
                            {
                                if (targetName[fileIndex] == "sqldb")
                                {
                                    fileEnd = resultLoc[i] + sqldbSearchLength(ref buffer, resultLoc[i]);
                                    if (fileEnd > buffer.Length)
                                        fileEnd = 0;
                                    if (buffer[fileEnd] != 0x38 && buffer[fileEnd + 1] != 0x38 && buffer[fileEnd + 1] != 0x3B)
                                    {
                                        RecordFileLocation(buffer, count, fileIndex, resultLoc[i], fileEnd, "");
                                        break;
                                    }
                                    else
                                        fileEnd = 0;
                                }
                                else
                                {
                                    if ((int)resultID[j] == footerType)
                                    {
                                        fileEnd = resultLoc[j] + targetEnd[fileIndex].Length;
                                        fileEnd = footerAdjust(fileEnd, targetName[fileIndex]);
                                        if (fileEnd > buffer.Length)
                                            fileEnd = 0;
                                        if (buffer[fileEnd] != 0x38 && buffer[fileEnd + 1] != 0x38 && buffer[fileEnd + 1] != 0x3B)
                                        {
                                            RecordFileLocation(buffer, count, fileIndex, resultLoc[i], fileEnd, "");
                                            break;
                                        }
                                        else
                                            fileEnd = 0;
                                    }
                                }
                            }
                        }

                        if (fileEnd == 0 && nextHeaderFound)
                            RecordFileLocation(buffer, count, fileIndex, resultLoc[i], nextHeader, "fragmented");
                        else if (fileEnd == 0 && !nextHeaderFound && searchRange != buffer.Length)
                            RecordFileLocation(buffer, count, fileIndex, resultLoc[i], Math.Min(resultLoc[i] + fileLength, buffer.Length), "partial");

                    }
                    else
                    {
                        RecordFileLocation(buffer, count, fileIndex, resultLoc[i], 0, "(non-carvable)");
                    }
                }
                i++;
            }

            updateFound();
        }

        // Records file location information from information passed by processing threads.
        private void RecordFileLocation(byte[] buffer, double count, int fileIndex, int start, int finish, string tag)
        {
            if (finish != 0)
            {
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

                resultRecord newEntry = new resultRecord((count + start), (count + finish), fileSize, sizeFormat, tag, targetName[fileIndex]);
                foundResults.Add(newEntry);

                Interlocked.Increment(ref results[fileIndex]);
            }
            else
            {
                resultRecord newEntry = new resultRecord((count + start), tag, targetName[fileIndex]);
                foundResults.Add(newEntry);

                Interlocked.Increment(ref results[fileIndex]);
            }
        }

        // Main file carving thread. Buffer is divided between logical cores assigned to file carve.
        private void carveResults(dataReader dataread)
        {
            updateHeader("Extracting files from data...");
            carveProcessed = 0;
            //double timez = MeasureTime(() =>
            //{
                Parallel.For(0, lpCount, i =>
                {
                    carveThread(i, ref dataread);
                });
                Task.WaitAll();
            //});

            //MessageBox.Show(timez.ToString());
        }

        private void carveThread(int cpu, ref dataReader dataread)
        {
            int currentFile = 0;
            while ((currentFile = Interlocked.Increment(ref carveProcessed)) <= carvableFiles.Count)
            {
                if (carvableFiles[currentFile-1].end != 0)
                {
                    updateGPUAct(cpu, true, true);
                    string filePath = saveLocation + carvableFiles[currentFile-1].filetype + "/";
                    if (!Directory.Exists(filePath))
                        Directory.CreateDirectory(filePath);
                    if (!File.Exists(filePath + carvableFiles[currentFile-1].start.ToString() + "." + carvableFiles[currentFile-1].filetype))
                    {
                        try
                        {
                            byte[] fileData = dataread.RetrieveFile((long)carvableFiles[currentFile-1].start, (long)carvableFiles[currentFile-1].end);
                            File.WriteAllBytes(filePath + carvableFiles[currentFile-1].start.ToString() + carvableFiles[currentFile-1].tag + "." + carvableFiles[currentFile-1].filetype, fileData);
                        }
                        catch { }
                    }
                    updateGPUAct(cpu, false, true);
                }

                // Update progress
                double Progress = (double)Math.Round((((double)currentFile / carvableFiles.Count) * 100) / 10.0 * 10);
                updateProgress((int)Progress, currentFile, carvableFiles.Count);
            }
        }

        // Experimental sqldb Search
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

        // Footer adjust for tailed files.
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


        #region Carvable File Data Save/Load Functions

        private void saveCarvableLocations<T>(T serializableObject, string fileName)
        {
            if (serializableObject == null) { return; }

            try
            {
                XmlDocument xmlDocument = new XmlDocument();
                XmlSerializer serializer = new XmlSerializer(serializableObject.GetType());
                using (MemoryStream stream = new MemoryStream())
                {
                    serializer.Serialize(stream, serializableObject);
                    stream.Position = 0;
                    xmlDocument.Load(stream);
                    xmlDocument.Save(fileName);
                    stream.Close();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not save carvable file locations!\nError: " + ex, "Save Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private T loadCarvableLocations<T>(string fileName)
        {
            if (string.IsNullOrEmpty(fileName)) { return default(T); }

            T objectOut = default(T);

            try
            {
                XmlDocument xmlDocument = new XmlDocument();
                xmlDocument.Load(fileName);
                string xmlString = xmlDocument.OuterXml;

                using (StringReader read = new StringReader(xmlString))
                {
                    Type outType = typeof(T);

                    XmlSerializer serializer = new XmlSerializer(outType);
                    using (XmlReader reader = new XmlTextReader(read))
                    {
                        objectOut = (T)serializer.Deserialize(reader);
                        reader.Close();
                    }

                    read.Close();
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show("Could not load carvable file locations!\nError: " + ex, "Load Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            return objectOut;
        }

        #endregion

        // Clean up before closing form
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
