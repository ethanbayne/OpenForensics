using System;
using System.Collections.Generic;
using System.Linq;
using Cudafy;
using Cudafy.Host;
using Cudafy.Atomics;
using Cudafy.Translator;

namespace OpenForensics
{
    public class Engine
    {
        private GPGPU gpu;
        private GPGPUProperties prop;
        public int gpuCoreCount;
        public int gpuBlocks;
        public int gpuOperatingCores;
        public int blockThreads;
        public int blockSize;
        public uint chunkSize;

        private uint bufferSize;
        private int initialState;
        private int fileLength;
        private int longestTarget;
        private byte[][] resultLoc;
        private int[][] resultCount;
        private byte[][] targetEnd;

        private bool carveOp;

        private byte[][] dev_buffer;
        private byte[][] dev_target;
        private int[,] dev_lookup;
        private byte[][] dev_targetEnd;
        private byte[][] dev_resultLoc;
        private int[][] dev_resultCount;


        #region Engine Initiation

        // Brute Force Engine
        public Engine(int GPUid, byte[][] target, byte[][] targetEnd, int fileLength, uint size, bool carveOp)
        {
            gpu = CudafyHost.GetDevice(CudafyModes.Target, GPUid);
            gpu.FreeAll();
            LoadModule();

            prop = gpu.GetDeviceProperties();
            chunkSize = size;
            // Find out maximum GPU Blocks and Supported Threads for each Block
            gpuBlocks = prop.WarpSize;
            blockThreads = prop.MaxThreadsPerBlock;
            blockSize = Math.Min(blockThreads, (int)Math.Ceiling(chunkSize / (float)blockThreads));  //Find the optimum size of the threads to handle the buffer

            this.fileLength = fileLength;

            // Copy target array to GPU for analysis
            dev_target = new byte[target.Length][];
            for (int i = 0; i < target.Length; i++)
                dev_target[i] = gpu.CopyToDevice<byte>(target[i]);


            // Allocate the memory on the GPU for buffer and results
            dev_buffer = new byte[gpuCoreCount][];
            dev_resultLoc = new byte[gpuCoreCount][];
            dev_resultCount = new int[gpuCoreCount][];
            resultCount = new int[gpuCoreCount][];
            for (int i = 0; i < gpuCoreCount; i++)
            {
                dev_buffer[i] = gpu.Allocate<byte>(new byte[chunkSize]);
                dev_resultLoc[i] = gpu.Allocate<byte>(new byte[chunkSize]);
                gpu.Set(dev_resultLoc[i]);
                dev_resultCount[i] = gpu.Allocate<int>(new int[target.Length]);
                gpu.Set(dev_resultCount[i]);
                resultCount[i] = new int[target.Length];
            }
            bufferSize = chunkSize;


            this.carveOp = carveOp;

            if (carveOp)
            {
                this.targetEnd = targetEnd;

                // Copy target array to GPU for analysis
                dev_targetEnd = new byte[targetEnd.Length][];
                for (int i = 0; i < targetEnd.Length; i++)
                    dev_targetEnd[i] = gpu.CopyToDevice<byte>(targetEnd[i]);
            }
        }


        // PFAC Algorithm Engine
        public Engine(int GPUid, int coreCount, byte[][] target, int[][] lookup, int longestTarget, int fileLength, uint size, bool carveOp)
        {
            gpu = CudafyHost.GetDevice(CudafyModes.Target, GPUid);
            gpu.SetCurrentContext();
            gpu.FreeAll();
            LoadModule();
            
            gpu.EnableMultithreading();
            for (int i = 0; i < gpuCoreCount; i++)
                gpu.CreateStream(i);

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

            this.carveOp = carveOp;

            // Allocate the memory on the GPU for buffer and results
            dev_buffer = new byte[gpuCoreCount][];
            dev_resultLoc = new byte[gpuCoreCount][];
            dev_resultCount = new int[gpuCoreCount][];
            resultLoc = new byte[gpuCoreCount][];
            resultCount = new int[gpuCoreCount][];
            for (int i = 0; i < gpuCoreCount; i++)
            {
                dev_buffer[i] = gpu.Allocate<byte>(new byte[chunkSize]);
                dev_resultLoc[i] = gpu.Allocate<byte>(new byte[chunkSize]);
                dev_resultCount[i] = gpu.Allocate<int>(new int[target.Length]);
                resultLoc[i] = new byte[chunkSize];
                resultCount[i] = new int[target.Length];
                FreeBuffers(i);
            }
        }

        #endregion


        #region CPU Operations

        public static int[] CPUPFACAnalyse(bool carveOp, byte[] buffer, int[][] lookup, byte[] resultsLoc, int numTargets, int longestTarget, int fileLength, int initialState, ref int[] rewindCheck)
        {
            int[] results = new int[numTargets];
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
                        if (!carveOp || state % 2 != 0)
                            results[state - 1]++;
                        if (carveOp)
                            resultsLoc[i] = (byte)state;
                    }
                    pos++;
                }
            }

            return results;
        }

        public static int[] CPUPFACSearch(byte[] buffer, int[][] lookup, int initialState, int fileLength, int offset, List<string> targetName)
        {
            int i = offset;
            int searchRange = buffer.Length;
            int n = buffer.Length;
            bool overRange = false;

            int headerType = 0;
            int footerType = 0;
            int fileIndex = 0;
            int startLocation = 0;
            int endLocation = 0;
            int rewind = 0;

            int state;
            int pos;

            for (; i < n; i++)                    // Loop to scan full file segment
            {
                state = initialState;
                pos = i;

                while (pos < n && startLocation == 0)
                {
                    state = lookup[state][buffer[pos]];
                    if (state == 0) { break; }
                    if (state < initialState && state % 2 != 0)
                    {
                        headerType = state;
                        fileIndex = ((headerType + 1) / 2) - 1;
                        for (int j = 0; j < targetName.Count; j++)
                        {
                            if (targetName[j] == targetName[fileIndex])
                            {
                                footerType = (j * 2) + 2;
                                break;
                            }
                        }
                        startLocation = i;
                        break;
                    }

                    pos++;
                }
            }

            if (startLocation != 0)
            {
                i = startLocation + 1;
                searchRange = startLocation + fileLength;
                if (searchRange > buffer.Length)
                {
                    overRange = true;
                    searchRange = buffer.Length;
                }

                for (; i < searchRange; i++)                    // Loop to scan full file segment
                {
                    state = initialState;
                    pos = i;

                    while (pos < searchRange && endLocation == 0)
                    {
                        state = lookup[state][buffer[pos]];
                        if (state == 0) { break; }
                        if (state < initialState && state == footerType)
                        {
                            if (buffer[pos + 1] == 0x38 || buffer[pos + 2] == 0x38 || buffer[pos + 2] == 0x3B)
                                pos++;
                            else
                            {
                                endLocation = pos;
                                break;
                            }
                        }
                        pos++;
                    }
                }

                if (endLocation == 0 && overRange)
                    rewind = fileLength;
            }
            else
                startLocation = buffer.Length;

            int[] foundResult = new int[4] { startLocation, endLocation, rewind, fileIndex };

            return foundResult;
        }

        //CPU Analyse - using typical Boyer-Moore Algorithm for searching Bytes
        public static int[] CPUBMAnalyse(byte[] buffer, byte[][] target, int[][] lookup)
        {
            int[] results = new int[target.Length];

            for (int i = 0; i < target.Length; i++)
            {
                int index = target[i].Length - 1;      // Create an index
                byte lastByte = target[i].Last();

                while (index < buffer.Length)       // Loop to scan full file segment
                {
                    var checkByte = buffer[index];  // Store check byte as the current scanned byte

                    if (buffer[index] == lastByte)  // If the byte scanned is equal to the last byte of target
                    {
                        bool found = true;              // Assume it's found

                        for (int j = target[i].Length - 2; j >= 0; j--)                // Scan the rest of the target
                        {
                            if (buffer[index - target[i].Length + j + 1] != target[i][j]) // If rest does not match target
                            {
                                found = false;          // False hit
                                break;                  // Break loop
                            }
                        }

                        if (found)                      // If found is positive after scanning other bytes
                        {
                            results[i] += 1;
                            index++;
                        }
                        else                            // If not positive
                        {
                            index++;                    // Go to next byte
                        }
                    }
                    else                            // If last byte does not match target
                    {
                        index += lookup[i][checkByte]; // Go to next byte according to position of current byte
                    }
                }
            }

            return results;
        }

        //CPU Analyse - using typical Boyer-Moore Algorithm for searching Bytes
        public static int[] CPUBMSearch(bool header, byte[] buffer, byte[] target, int[] lookup, byte[] targetEnd, int[] lookupEnd, int fileLength, int offset)
        {
            int i = 0;
            int searchRange = buffer.Length;
            bool overRange = false;

            int startLocation = 0;
            int endLocation = 0;
            int rewind = 0;
    
            byte lastByte = target.Last();

            for (int h = 0; h < target.Length - 1; h++)
            {
                if (buffer[h] == lastByte)
                    if (rewind < target.Length)
                        rewind = target.Length + 1;
            }

            i = offset + target.Length;

            while (i < searchRange)       // Loop to scan full file segment
            {
                var checkByte = buffer[i];  // Store check byte as the current scanned byte

                if (buffer[i] == lastByte)  // If the byte scanned is equal to the last byte of target
                {
                    bool found = true;              // Assume it's found

                    for (int j = target.Length - 2; j >= 0; j--)                // Scan the rest of the target
                    {
                        if (buffer[i - target.Length + j + 1] != target[j]) // If rest does not match target
                        {
                            found = false;          // False hit
                            break;                  // Break loop
                        }
                    }

                    if (found)                      // If found is positive after scanning other bytes
                    {
                        startLocation = i - target.Length + 1;
                        break;
                    }
                    else                            // If not positive
                    {
                        i++;                    // Go to next byte
                    }
                }
                else                            // If last byte does not match target
                {
                    i += lookup[checkByte]; // Go to next byte according to position of current byte
                }
            }


            if (startLocation != 0)
            {
                searchRange = startLocation + fileLength;
                if (searchRange > buffer.Length)
                {
                    overRange = true;
                    searchRange = buffer.Length;
                }

                i = startLocation + targetEnd.Length;
                lastByte = targetEnd.Last();

                while (i < searchRange && endLocation == 0)       // Loop to scan full file segment
                {
                    var checkByte = buffer[i];  // Store check byte as the current scanned byte

                    if (buffer[i] == lastByte)  // If the byte scanned is equal to the last byte of target
                    {
                        bool found = true;              // Assume it's found

                        for (int j = targetEnd.Length - 2; j >= 0; j--)                // Scan the rest of the target
                        {
                            if (buffer[i - targetEnd.Length + j + 1] != targetEnd[j]) // If rest does not match target
                            {
                                found = false;          // False hit
                                break;                  // Break loop
                            }
                        }

                        if (found)                      // If found is positive after scanning other bytes
                        {
                            if (buffer[i + 1] == 0x38 || buffer[i + 2] == 0x38 || buffer[i + 2] == 0x3B)
                                i++;
                            else
                            {
                                endLocation = i + 1;
                                break;
                            }
                        }
                        else                            // If not positive
                        {
                            i++;                    // Go to next byte
                        }
                    }
                    else                            // If last byte does not match target
                    {
                        i += lookupEnd[checkByte]; // Go to next byte according to position of current byte
                    }
                }

                if (endLocation == 0 && overRange)
                    rewind = fileLength;
            }
            else
                startLocation = buffer.Length;

            int[] foundResult = new int[3] {startLocation, endLocation, rewind};

            return foundResult;
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
                if (carveOp)
                    gpu.Set(dev_resultLoc[gpuCore]);
                gpu.Set(dev_resultCount[gpuCore]);
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

        public byte[] ReturnResult(int gpuCore)
        {
            return resultLoc[gpuCore];

        }

        public int ReturnResultCount(int gpuCore, int c)
        {
            return resultCount[gpuCore][c];
        }

        public void LaunchBruteCarving(int gpuCore)
        {
            gpu.SetCurrentContext();

            byte fileID = 1;

            for (int i = 0; i < dev_target.Length; i++)
            {
                gpu.Launch(gpuBlocks, blockSize).BruteAnalyse(fileID, dev_buffer, dev_target[i], i, fileLength, 0, bufferSize, dev_resultLoc, dev_resultCount);  // Start the analysis of the buffer
                fileID += 1;
                if (carveOp)
                {
                    if (i == 0 || targetEnd[i].SequenceEqual(targetEnd[i - 1]) == false)
                        gpu.Launch(gpuBlocks, blockSize).BruteAnalyse(fileID, dev_buffer, dev_targetEnd[i], i, fileLength, 0, bufferSize, dev_resultLoc, dev_resultCount);  // Start the analysis of the buffer
                    fileID += 1;
                }
            }

            //for (int i = 0; i < (dev_target.Length + 2); i++)
            //    gpu.SynchronizeStream(i);
            gpu.Synchronize();
            if(carveOp)
                gpu.CopyFromDevice(dev_resultLoc[gpuCore], resultLoc[gpuCore]);                       // Copy results back from GPU
            gpu.CopyFromDevice(dev_resultCount[gpuCore], resultCount[gpuCore]);
            FreeBuffers(gpuCore);
        }

        public void LaunchPFACCarving(int gpuCore)
        {
            int isCarveOp = 0;
            if (carveOp)
                isCarveOp = 1;

            //int size = Marshal.SizeOf(buffer[0]) * buffer.Length;
            //IntPtr bufferPtr = gpu.HostAllocate<int>(size);
            //IntPtr resultPtr = gpu.HostAllocate<int>(size);
            //Marshal.Copy(bufferPtr, buffer, 0, buffer.Length);

            //gpu.CopyToDeviceAsync(bufferPtr, 0, dev_buffer, 0, size, 1);
            //gpu.LaunchAsync(gpuBlocks, blockSize, 1, "PFACAnalyse", dev_buffer, initialState, isCarveOp, dev_lookup, longestTarget, fileLength, dev_resultLoc, dev_resultCount);
            //gpu.CopyFromDeviceAsync(dev_resultLoc, 0, resultPtr, 0, size, 1);
            //gpu.SynchronizeStream(1);

            //GPGPU.CopyOnHost(resultPtr, 0, resultLoc, 0, size);
            //gpu.HostFree(bufferPtr);
            //gpu.HostFree(resultPtr);
            //gpu.DestroyStream(1);
            gpu.SetCurrentContext();

            gpu.LaunchAsync(gpuOperatingCores, blockSize, gpuCore, "PFACAnalyse", dev_buffer[gpuCore], initialState, isCarveOp, dev_lookup, longestTarget, fileLength, dev_resultLoc[gpuCore], dev_resultCount[gpuCore]);
            //gpu.Launch(gpuOperatingCores, blockSize, gpuCore).PFACAnalyse(dev_buffer[gpuCore], initialState, isCarveOp, dev_lookup, longestTarget, fileLength, dev_resultLoc[gpuCore], dev_resultCount[gpuCore]);  // Start the analysis of the buffer
            gpu.SynchronizeStream(gpuCore);
            gpu.CopyFromDevice(dev_resultCount[gpuCore], resultCount[gpuCore]);
            if (carveOp)
                for (int i = 0; i < resultCount[gpuCore].Length; i++)
                {
                    if (resultCount[gpuCore][i] > 0)
                    {
                        gpu.CopyFromDevice(dev_resultLoc[gpuCore], resultLoc[gpuCore]);                       // Copy results back from GPU
                        break;
                    }
                }

            //gpu.Synchronize();
            FreeBuffers(gpuCore);
        }

        //GPU Analyse - using brute force for searching Bytes
        [Cudafy]
        public static void PFACAnalyse(GThread thread, byte[] buffer, int initialState, int carveOp, int[,] lookup, int longestTarget, int fileLength, byte[] results, uint[] resultCount)
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
                        if (carveOp == 0)
                            thread.atomicAdd(ref resultCount[state - 1], 1);
                        else if (carveOp == 1)
                        {
                            if((state - 1) % 2 == 0)
                                thread.atomicAdd(ref resultCount[(int)(state / 2) - 1], 1);
                            results[i] = (byte)state;
                        }
                    }
                    pos++;
                }
            }

            thread.SyncThreads();                                                   // Sync GPU threads
        }


        //GPU Analyse - using brute force for searching Bytes
        [Cudafy]
        public static void BruteAnalyse(GThread thread, byte header, byte[] buffer, byte[] target, int targetNo, int fileLength, int offset, uint searchRange, byte[] results, uint[] resultCount)
        {
            int i = thread.threadIdx.x + thread.blockIdx.x * thread.blockDim.x;     // Counter for i
            int stride = thread.blockDim.x * thread.gridDim.x;                      // Stride is the next byte for the thread to go to

            uint[] temp = thread.AllocateShared<uint>("temp", 1024);                // Allocate shared memory for storing results
            temp[thread.threadIdx.x] = 0;                                           // Set shared memory to 0
            thread.SyncThreads();                                                   // Sync GPU threads

            while (i < (searchRange - (target.Length + 1)))                    // Loop to scan full file segment
            {
                if (buffer[i] == target[0])      // If the byte scanned is equal to the first byte of target
                {
                    bool found = true;              // Assume it's found

                    for (int j = 1; j <= target.Length - 1; j++)            // Scan the rest of the target
                    {
                        if (buffer[i + j] != target[j]) // If rest does not match target
                        {
                            found = false;          // False hit
                            break;                  // Break loop
                        }
                    }

                    if (found)                      // If found is positive after scanning other bytes
                    {
                        temp[thread.threadIdx.x] += 1;  // Found match, plus 1 to results
                        results[i] = header;
                        i += stride;
                    }
                    else                            // If not positive
                    {
                        i += stride;                    // Go to next byte
                    }
                }
                else                            // If last byte does not match target
                {
                    i += stride; // Go to next byte according to position of current byte
                }
            }

            thread.SyncThreads();                                                   // Sync GPU threads
            thread.atomicAdd(ref resultCount[targetNo], temp[thread.threadIdx.x]);      // Add up results each thread produced
        }

        #endregion


        #region Target Operations

        public static int[][] pfacLookupCreate(Byte[][] target)
        {
            int[] idx = new int[target.Length];     // Create idx to index targets
            for (int i = 0; i < idx.Length; i++)
                idx[i] = i;

            Array.Sort<int>(idx, (a, b) => target[a].Length.CompareTo(target[b].Length));       // Sort targets by element length, using idx to index shortest to longest target

            List<int[]> table = new List<int[]>();
            table.Add(new int[256]);                   // Create Fail State row (row 0)

            for (int i = 0; i < target.Length; i++)     // Create blank line for each target to signify successful state (will be populated later)
                table.Add(new int[256]);

            table.Add(new int[256]);                   // Start Point row (index of target length + 1)

            int walkIndex = target.Length + 1;          // "walkIndex" pointer will keep an index of where we are in the table as we traverse with new targets to be added
            int state = walkIndex + 1;                  // "state" keeps a pointer on where the next unused row/state should be created

            for (int i = 0; i < target.Length; i++)     // Populate table with targets
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
                    if(table.Count <= walkIndex)                        // If the table doesn't have enough rows for any new states, create one
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

            int[][] result = table.ToArray();       // Convert list to 2d array for returning

            /* Logic Table Generation to CSV File
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


        public static int[][] bmLookupCreate(Byte[][] target)
        {
            int[][] lookup = new int[target.Length][];          // Calculate Lookup reference for Boyer-Moore algorithm
            for (int i = 0; i < lookup.Length; i++)             // Do for each Search Target
            {
                lookup[i] = new int[256];
                for (int j = 0; j < lookup[i].Length; j++)
                {
                    lookup[i][j] = target[i].Length;
                }
            }

            for (int i = 0; i < lookup.Length; i++)
            {
                for (int j = 0; j < target[i].Length; j++)
                {
                    lookup[i][target[i][j]] = target[i].Length - j - 1;
                }
            }

            return lookup;
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
