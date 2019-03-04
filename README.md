![alt text](https://github.com/ethanbayne/OpenForensics/blob/master/OpenForensics/Resources/OpenForensicsLogo2.png "OpenForensics Logo")

OpenForensics is an open-source OpenCL Digital Forensics analysis and file carving tool. This prototype tool was built in conjunction with PhD research from Dr Ethan Bayne as a platform to demonstrate the performance enhancements possible when applying GPGPU processing and an optimised PFAC algorithm to the problem of string searching in the field of Digital Forensics. Details of the supporting research can be found in the completed thesis by Dr Ethan Bayne:

https://rke.abertay.ac.uk/en/studentTheses/accelerating-digital-forensic-searching-through-gpgpu-parallel-pr

This prototype tool was built using C# and Cudafy.NET (https://cudafy.codeplex.com/). Currently, the file carving provided by this tool is considered basic (carves data between found header and footer). Whilst it is intended to develop more advanced file carving features in time, we would welcome collaborators to help build upon the functionality of the tool.

# Basic Usage
   1. Provide a case reference and evidence reference ___(Optional)___.
   2. Select either a physical drive or a file to perform physical searching on.
   3. Select targets _(either file-types or keywords)_.
   4. Choose processing technique ___(Optional - GPU usage recommended)___.
   5. Choose whether you would like to generate a live preview of JPG images found during searching. ___(Optional - Searching will be slower and will cause higher memory usage with this enabled)___.
   6. Click Analyse.
   7. Choose location to save results.

OpenForensics will create a set of sub-folders and files to store results. The directory and file hierarchy created is: 

    <save-location>

      └─ <case-reference> default:"OpenForensics Output"
 
        └─ <evidence-reference> default:file/drive name
            ├─ LogFile.txt
            ├─ CarvableFileData.of
            └─ <Folder of each file-type carved>
                └─ <Files carved>

If data exists for the case and file/storage device, an option will present itself to query whether you want to reproduce files using existing results. Selecting No will prompt whether you want to overwrite the existing results with a new search. Warning: Old results will be erased when proceeding.

File-types are populated from the FileTypes.xml file. OpenForensics will load the file-types specified in this file when the program launches. Each file-type specified in FileTypes.xml should have values for:

   1. __Type__ - _(Image/Video/Audio/Document/Misc)_ - used to categorise file types for batch searching.
   2. __Name__ - Used as a file identifier and also for file extension for carved files.
   3. __Value__ - File header in byte format without spaces _(e.g. FFD8FFE00010)_.
   4. __EOF__ ___(Optional - Required for file carving)___ - File footer in byte format without spaces _(e.g. FFD9)_.
   5. __MaxLengthMB__ ___(Optional)___ - Specifies the maximum file length for the type of file _(Default: 10 MiB)_.

# Open-source License
OpenForensics Copyright © 2016 - 2019 Ethan Bayne

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

This software includes CUDAfy .NET (Cudafy.NET.dll), developed by Hybrid DSP.

The LGPL v2.1 License applies to CUDAfy .NET. If you wish to modify the code then changes should be re-submitted to Hybrid DSP. If you wish to incorporate Cudafy.NET into your own application instead of redistributing the dll's then please consider a commerical license. Visit http://www.hybriddsp.nl. This will also provide you with priority support and contribute to on-going development.

Cudafy.NET also utilises the following libraries are made use of:
The MIT license applies to ILSpy, NRefactory and ICSharpCode.Decompiler (Copyright (c) 2011 AlphaSierraPapa for the SharpDevelop team).
Mono.Cecil also uses the MIT license (Copyright JB Evain).
CUDA.NET is a free for use license (Copyright Company for Advanced Supercomputing Solutions Ltd)
