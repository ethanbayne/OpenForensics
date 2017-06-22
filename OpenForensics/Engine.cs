using System;
using System.Collections;
using System.Collections.Generic;
using Cudafy;
using Cudafy.Host;
using Cudafy.Atomics;
using Cudafy.Translator;

namespace OpenForensics
{

    public class Engine
    {
        private int GPUid;
        private GPGPU gpu;
        private GPGPUProperties prop;
        public int gpuCoreCount;
        public int gpuBlocks;
        public int gpuOperatingCores;
        public int blockThreads;
        public int blockSize;
        public uint chunkSize;
        private uint resultCache = 1048576;

        private uint bufferSize;
        private int initialState;
        private int fileLength;
        private int longestTarget;
        private int[][] resultCount;

        private static object[] gpuThreadLock;
        private byte[][] dev_buffer;
        private int[,] dev_lookup;
        private int[][] dev_resultCount;

        private byte[][] foundID;
        private int[][] foundLoc;
        private int[][] dev_foundCount;
        private byte[][] dev_foundID;
        private int[][] dev_foundLoc;

        #region Engine Initiation

        // PFAC Algorithm Engine
        public Engine(int GPUid, int coreCount, byte[][] target, int[][] lookup, int longestTarget, int fileLength, uint size)
        {
            this.GPUid = GPUid;
            gpu = CudafyHost.GetDevice(CudafyModes.Target, GPUid);
            gpu.SetCurrentContext();
            gpu.FreeAll();
            LoadModule();
            
            gpu.EnableMultithreading();
            for (int i = 0; i < gpuCoreCount; i++)
                gpu.CreateStream(i);

            gpuThreadLock = new object[coreCount];
            for (int i = 0; i < gpuThreadLock.Length; i++)
                gpuThreadLock[i] = new Object();

            prop = gpu.GetDeviceProperties();
            gpuCoreCount = coreCount;
            chunkSize = size;
            // Find out maximum GPU Blocks and Supported Threads for each Block
            gpuBlocks = prop.WarpSize;
            gpuOperatingCores = gpuBlocks;
            //gpuOperatingCores = gpuBlocks / gpuCoreCount; // This is fine for nVidia... everything else dies horribly!
            //if (gpuOperatingCores == 0)
            //    gpuOperatingCores = 1;
            blockThreads = prop.MaxThreadsPerBlock;
            blockSize = Math.Min(blockThreads, (int)Math.Ceiling(chunkSize / (float)blockThreads));  //Find the optimum size of the threads to handle the buffer

            //MessageBox.Show("GPU Blocks: " + gpuBlocks.ToString() + Environment.NewLine + "Block Threads: " + blockThreads.ToString() + Environment.NewLine + "Block Size: " + blockSize.ToString());

            this.fileLength = fileLength;
            this.longestTarget = longestTarget;

            int[,] newLookup = new int[lookup.Length, 256];
            for (int x = 0; x < lookup.Length; x++)
                for (int y = 0; y < 256; y++)
                    newLookup[x, y] = lookup[x][y];
            dev_lookup = new int[lookup.Length, 256];
            dev_lookup = gpu.CopyToDevice<int>(newLookup);

            initialState = target.Length + 1;
            bufferSize = chunkSize;

            // Allocate the memory on the GPU for buffer and results
            dev_buffer = new byte[gpuCoreCount][];
            dev_resultCount = new int[gpuCoreCount][];
            resultCount = new int[gpuCoreCount][];

            dev_foundCount = new int[gpuCoreCount][];
            dev_foundID = new byte[gpuCoreCount][];
            dev_foundLoc = new int[gpuCoreCount][];
            foundID = new byte[gpuCoreCount][];
            foundLoc = new int[gpuCoreCount][];

            for (int i = 0; i < gpuCoreCount; i++)
            {
                dev_buffer[i] = gpu.Allocate<byte>(new byte[chunkSize]);
                dev_resultCount[i] = gpu.Allocate<int>(new int[target.Length]);
                resultCount[i] = new int[target.Length];

                dev_foundCount[i] = gpu.Allocate<int>(new int[1]);
                dev_foundID[i] = gpu.Allocate<byte>(new byte[resultCache]);
                dev_foundLoc[i] = gpu.Allocate<int>(new int[resultCache]);
                foundID[i] = new byte[resultCache];
                foundLoc[i] = new int[resultCache];
                FreeBuffers(i);
            }
        }

        #endregion


        #region CPU Operations

        //CPU PFAC Analyse - using PFAC Algorithm for searching Bytes
        public static int[] CPUPFACAnalyse(byte[] buffer, int[][] lookup, ref byte[] resultsID, ref int[] resultsLoc, int numTargets)
        {
            int[] results = new int[numTargets];
            int initialState = numTargets + 1;
            int foundCount = 0;
            int n = buffer.Length;

            for (int i = 0; i < n; i ++)                    // Loop to scan full file segment
            {
                int state = initialState;
                int pos = i;

                while (pos < n)
                {
                    state = lookup[state][buffer[pos]];
                    if (state == 0) { break; }
                    if (state < initialState)
                    {
                        if (state % 2 != 0)
                        {
                            results[((state + 1) / 2) - 1]++;
                            resultsID[foundCount] = (byte)state;
                            resultsLoc[foundCount] = i;
                            foundCount++;
                        }
                    }
                    pos++;
                }
            }

            return results;
        }

        #endregion


        #region GPU Operations


        public string GetName()
        {
            return prop.DeviceId.ToString() + ": " + prop.Name.Trim();
        }

        public void CopyToDevice(int gpuCore, byte[] buffer)
        {
            bufferSize = (uint)buffer.Length;
            gpu.SetCurrentContext();
            gpu.CopyToDevice<byte>(buffer, dev_buffer[gpuCore]);
        }

        public void FreeBuffers(int gpuCore)
        {
            try
            {
                gpu.SetCurrentContext();
                gpu.Set(dev_buffer[gpuCore]);
                gpu.Set(dev_resultCount[gpuCore]);

                gpu.Set(dev_foundCount[gpuCore]);
                gpu.Set(dev_foundID[gpuCore]);
                gpu.Set(dev_foundLoc[gpuCore]);
            }
            catch { }
        }

        public void FreeAll()
        {
            try
            {
                gpu.SetCurrentContext();
                gpu.FreeAll();
            }
            catch { }
        }

        public void HostFreeAll()
        {
            try
            {
                gpu.SetCurrentContext();
                gpu.HostFreeAll();
            }
            catch { }
        }

        public void LoadModule()
        {
            CudafyModule km = CudafyModule.TryDeserialize();    // Look for cdfy module file before generating

            // Ensure if using Cuda, use 2.0 architecture for Atomics compatibility
            if (km == null || !km.TryVerifyChecksums())
            {
                km = CudafyTranslator.Cudafy();
            }
            gpu.LoadModule(km);
        }

        public byte[] ReturnResultID(int gpuCore)
        {
            return foundID[gpuCore];

        }
        public int[] ReturnResultLoc(int gpuCore)
        {
            return foundLoc[gpuCore];

        }

        public int ReturnResultCount(int gpuCore, int c)
        {
            return resultCount[gpuCore][c];
        }

        //GPU PFAC Carving - using PFAC for searching Bytes
        public void LaunchPFACCarving(int gpuCore)
        {

            lock (gpuThreadLock[GPUid])
            {
                gpu.SetCurrentContext();

                gpu.LaunchAsync(gpuOperatingCores, blockSize, gpuCore, "PFACAnalyse", dev_buffer[gpuCore], initialState, dev_lookup, dev_resultCount[gpuCore], dev_foundCount[gpuCore], dev_foundID[gpuCore], dev_foundLoc[gpuCore]);
                //gpu.Launch(gpuOperatingCores, blockSize, gpuCore).PFACAnalyse(dev_buffer[gpuCore], initialState, dev_lookup, longestTarget, fileLength, dev_resultLoc[gpuCore], dev_resultCount[gpuCore]);  // Start the analysis of the buffer
                gpu.SynchronizeStream(gpuCore);
            }

            gpu.CopyFromDevice(dev_resultCount[gpuCore], resultCount[gpuCore]);

            for (int i = 0; i < resultCount[gpuCore].Length; i++)
            {
                if (resultCount[gpuCore][i] > 0)
                {
                    gpu.CopyFromDevice(dev_foundID[gpuCore], foundID[gpuCore]);
                    gpu.CopyFromDevice(dev_foundLoc[gpuCore], foundLoc[gpuCore]);
                    break;
                }
            }

            //gpu.Synchronize();
            FreeBuffers(gpuCore);
        }

        //GPU PFAC Analyse - using PFAC for searching Bytes
        [Cudafy]
        public static void PFACAnalyse(GThread thread, byte[] buffer, int initialState, int[,] lookup, uint[] resultCount, int[] foundCount, byte[] foundID, int[] foundSOF)
        {
            int n = buffer.Length;

            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;     // Counter for i
            int stride = thread.blockDim.x * thread.gridDim.x;                      // Stride is the next byte for the thread to go to

            for (; i < n; i += stride)                    // Loop to scan full file segment
            {
                int state = initialState;
                int pos = i;

                while (pos < n)
                {
                    state = lookup[state,buffer[pos]];
                    if (state == 0) { break; }
                    if (state < initialState)
                    {
                        if ((state - 1) % 2 == 0)
                            thread.atomicAdd(ref resultCount[(int)((state + 1) / 2) - 1], 1);

                        int counter = thread.atomicAdd(ref foundCount[0], 1);
                        foundID[counter] = (byte)state;
                        foundSOF[counter] = i;
                    }
                    pos++;
                }
            }

            thread.SyncThreads();                                                   // Sync GPU threads
            
        }

        #endregion


        #region Target Operations

        public static int[][] pfacLookupCreate(Byte[][] target)
        {
            int[] idx = new int[target.Length];     // Create idx to index targets
            for (int i = 0; i < idx.Length; i++)
                idx[i] = i;

            ArrayList listofNull = new ArrayList();
            for (int i = 0; i < target.Length; i++)
                if (target[i] == null)
                {
                    listofNull.Add(i);
                    target[i] = new byte[] { 0 };
                }

            Array.Sort<int>(idx, (a, b) => target[a].Length.CompareTo(target[b].Length));       // Sort targets by element length, using idx to index shortest to longest target

            for (int i = 0; i < listofNull.Count; i++)
                target[(int)listofNull[i]] = null;
            listofNull.Clear();

            List<int[]> table = new List<int[]>();
            table.Add(new int[256]);                   // Create Fail State row (row 0)

            for (int i = 0; i < target.Length; i++)     // Create blank line for each target to signify successful state (will be populated later)
                table.Add(new int[256]);

            table.Add(new int[256]);                   // Start Point row (index of target length + 1)

            int walkIndex = target.Length + 1;          // "walkIndex" pointer will keep an index of where we are in the table as we traverse with new targets to be added
            int state = walkIndex + 1;                  // "state" keeps a pointer on where the next unused row/state should be created

            for (int i = 0; i < target.Length; i++)     // Populate table with targets
            {
                if (target[idx[i]] != null)
                {
                    walkIndex = target.Length + 1;          // Set walk pointer to start point

                    if (table[walkIndex][target[idx[i]][0]] == 0)        // Check to see if first character already has a path, if not, set a new state to traverse to
                    {
                        table[walkIndex][target[idx[i]][0]] = state;
                        state++;
                    }

                    walkIndex = table[walkIndex][target[idx[i]][0]];    // Follow new, or existing state

                    for (int j = 1; j < target[idx[i]].Length; j++)     // For the rest of the characters of the target:
                    {
                        if (table.Count <= walkIndex)                        // If the table doesn't have enough rows for any new states, create one
                            table.Add(new int[256]);

                        if (table[walkIndex][target[idx[i]][j]] == 0)       // Check to see if character already has a path, if not, set a new state to traverse to
                        {
                            if (j != target[idx[i]].Length - 1)                 // If it's not the last character, use a new state..
                            {
                                table[walkIndex][target[idx[i]][j]] = state;
                                state++;
                            }
                            else                                                // Else, set the state to the row of the target index
                                table[walkIndex][target[idx[i]][j]] = idx[i] + 1;
                        }
                        else if (j == target[idx[i]].Length - 1)
                            if (i != 0 && (idx[i] + 1) % 2 == 0 && table[walkIndex][target[idx[i]][j]] > idx[i] + 1)
                                table[walkIndex][target[idx[i]][j]] = idx[i] + 1;

                        walkIndex = table[walkIndex][target[idx[i]][j]];    // Traverse to next state

                    }
                }
            }

            int[][] result = table.ToArray();       // Convert list to 2d array for returning

            // Logic Table Generation to CSV File for visualising logic table
            /*
            using (StreamWriter outfile = new StreamWriter(@"C:\Users\Ethan\Desktop\myfile.csv"))
            {
                for (int x = 0; x < result.Length; x++)
                {
                    string content = "";
                    for (int y = 0; y < result[x].Length; y++)
                    {
                        content += result[x][y].ToString();
                        if (y != result[x].Length - 1) 
                        content += ",";
                    }
                    outfile.WriteLine(content);
                }
            }
            */
            return result;
        }

        public static string StringtoHex(string keyword)
        {
            string hexOutput = "";
            foreach (char letter in keyword)
            {
                // Get the integral value of the character. 
                int value = Convert.ToInt32(letter);
                // Convert the decimal value to a hexadecimal value in string form. 
                hexOutput = hexOutput + String.Format("{0:X}", value);
            }
            return hexOutput;
        }

        public static byte[] GetBytes(string hex)
        {
            // Function to convert Hex to Bytes
            if (hex.Length % 2 == 1)
                throw new Exception("The Hex Value cannot have an odd number of digits. Validate XML Input.");

            byte[] bytes = new byte[hex.Length >> 1];

            for (int i = 0; i < hex.Length >> 1; ++i)
            {
                bytes[i] = (byte)((GetHexVal(hex[i << 1]) << 4) + (GetHexVal(hex[(i << 1) + 1])));
            }

            return bytes;
        }

        public static int GetHexVal(char hex)
        {
            // Function for getting Hex Value (Uppercase A-F)
            int val = (int)hex;
            return val - (val < 58 ? 48 : 55);
        }


        #endregion

    }
}
