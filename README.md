# OpenForensics
OpenForensics is an open-source OpenCL storage device and physical analysis and file carving tool. This tool was built in conjunction with PhD research from Dr Ethan Bayne as a platform to demonstrate the performance enhancements possible when applying GPGPU processing and an optimised PFAC algorithm to the problem of string searching in the field of Digital Forensics. Details of the supporting research can be found in the completed thesis by Dr Ethan Bayne:

<Thesis in review, link will be provided shortly>

This tool was built using C# and Cudafy.NET (https://cudafy.codeplex.com/). Currently, the file carving provided by this tool is considered basic (carves data between found header and footer). Whilst it is intended to develop more advanced file carving features in time, we would welcome collaborators to help build upon the functionality of the tool.

# Basic Usage
1. (Optional) Provide a case reference and evidence reference.
2. Select either a physical drive or a file to perform physical searching on.
3. Select targets (either file-types or keywords).
4. (Optional) Choose processing technique.
5. Click Analyse.

OpenForensics will create a set of sub-folders to store results. The directory hierarchy is: 

save-location

  └─case-reference default:"OpenForensics Output"
 
    └─evidence-reference default:file/drive name

If data exists for the case and file/storage device, an option will present itself to query whether you want to reproduce files using existing results. Selecting No will prompt whether you want to overwrite the existing results with a new search. Warning: Old results will be erased when proceeding.

# Open-source License
OpenForensics Copyright 2017 Ethan Bayne

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at:

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

---

The LGPL v2.1 License applies to CUDAfy .NET. If you wish to modify the code then changes should be re-submitted to Hybrid DSP. If you wish to incorporate Cudafy.NET into your own application instead of redistributing the dll's then please consider a commerical license. Visit http://www.hybriddsp.com. This will also provide you with priority support and contribute to on-going development.

Cudafy.NET also utilises the following libraries are made use of:
The MIT license applies to ILSpy, NRefactory and ICSharpCode.Decompiler (Copyright (c) 2011 AlphaSierraPapa for the SharpDevelop team).
Mono.Cecil also uses the MIT license (Copyright JB Evain).
CUDA.NET is a free for use license (Copyright Company for Advanced Supercomputing Solutions Ltd)
