using System;
using System.Collections.Generic;
using System.IO;
using System.Diagnostics;
using System.Threading;
using System.Windows.Forms;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Xml;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;
using System.Management;

namespace OpenForensics
{
    public partial class OpenForensics : Form
    {
        public OpenForensics()
        {
            InitializeComponent();
        }

        // Version 1.02b - Introduced memory cache to store end of segment to reduce flushing storage device cache.
        // Version 1.05b - Introduction of queued file reading to make optimal use of storage drive I/O.
        // Version 1.10b - Files are only carved by the CPU if there are *actually* files to carve.
        // Version 1.15b - Read queues tweaks for optimal storage device I/O. Fixed GPU PFAC searching where result could be part of a larger pattern, but terminated early.
        // Version 1.16b - Introduced queue limiter to stop over-queuing instructions and overwhelming slower drives.
        // Version 1.21b - Introduced GPU threading.
        // Version 1.22b - Fixed File carving after 1.21b patch. Introduced more GPU optimisations.
        // Version 1.23b - Small analysis interface change to cater for larger amounts of processors.
        // Version 1.24b - Fixed small logic bug with GPU carving function - thread.atomicAdd(ref resultCount[(int)(state / 2) - 1], 1); > thread.atomicAdd(ref resultCount[(int)((state + 1) / 2) - 1], 1);
        // Version 1.25b - Incremental refactoring of code.
        // Version 1.30b - Fixed non-Nvidia GPU flaw where multiple instanced use of GPU was mishandled. Implemented GPU locker so that only one thread can utilise the GPU at any given moment.
        // Version 1.50b - Overhaul and major refactoring of program. Optimised GPU result recording and significantly reduced CPU result processing. 
        // Version 1.51b - Fixed bugs (processing result method bug when threads > 1). Optimised result preparation.
        // Version 1.53b - .NET Framework v.4.5, introduced Async refinements to main CPU and GPU processing threads
        // Version 1.54b - Bug fix for Ryzen processors being counted as GPUs
        // Version 1.60b - Enhanced processing framework. Introduced post-processing stage after patterns found. Enabled window to be size of pattern rather than file. Transferred jpg checks to search processing. Corrected file reproduction technique.
        // Version 1.61b - Introduced file length setting in XML (default 10 MiB). Able to set different combinations of headers and footers for the same filetype by using the name format <filetype>-<identifier> (gif-2 provided in default XML as an example).
        // Version 1.7b - Introduced visualised analysis. Displays jpg images over 100KB during searching.
        // Version 1.71b - Analysis message box prompt replaced by buttons.
        // Version 1.73b - Analysis interface refactored for visualisation. Introduced thread-safe stop search function to safely abort searching.
        // Version 1.76b - Further refactoring and concurrency improvements with visualisation introduction.
        // Version 1.78b - Preparing visualisation branch merge into master by enabling optional image preview processing
        // Version 1.80b - Minor UI updates and improvements

        private string version = "v. 1.80b";   // VERSION INFORMATION TO DISPLAY

        private string TestType;             // Value for Platform Type Selected
        private bool multiGPU = false;
        private List<int> gpus = new List<int>();
        private List<long> gpuMem = new List<long>();
        private long maxGPUMem = 0;

        private string CaseName = "";
        private string EvidenceName = "";
        private string saveLocation = "";
        private string fileName = "";
        private string carvableFileRecord = "";

        private bool imagePreview = false;

        // Arrays for all search target types
        private List<string> targetName = new List<string>();
        private List<string> targetHeader = new List<string>();
        private List<string> targetFooter = new List<string>();
        private List<int> targetLength = new List<int>();

        private List<string> imageNames = new List<string>();
        private List<string> videoNames = new List<string>();
        private List<string> audioNames = new List<string>();
        private List<string> documentNames = new List<string>();
        private List<string> miscNames = new List<string>();


        private void OpenForensics_Load(object sender, EventArgs e)
        {
            lblVersion.Text = version;
            OFTooltips();

            try
            {
                Setup();                    // Attempt Initial GPGPU Interface Setup & Populate Formats
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.ToString(), "Setup Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Setup()
        {
            //Process.GetCurrentProcess().PriorityClass = ProcessPriorityClass.High;  // Sets Priority of Process to High
            PopulateGPGPUComboBox();        // Populate GPGPU Selection Box
            PopulateFileTypes();         // Get File Types from XML
            cboFileType.SelectedIndex = 0;
            cboKeywords.SelectedIndex = 0;  // Set Keywords to first entry on list
            TargetTypeUpdate();             // Update Target Type
            if (txtFile.Text != "")
                if (txtFile.Text.StartsWith("\\\\.\\"))
                    btnDriveOpen.BackColor = Color.DarkSeaGreen;
                else
                    btnFileOpen.BackColor = Color.DarkSeaGreen;
        }

        protected override void OnPaintBackground(PaintEventArgs e)
        {
            if (this.ClientRectangle.Width > 0 && this.ClientRectangle.Height > 0)
            {
                using (LinearGradientBrush brush = new LinearGradientBrush(this.ClientRectangle, Color.White, Color.Black, 90F))
                {
                    e.Graphics.FillRectangle(brush, this.ClientRectangle);
                }
            }
        }


        #region Interface Interactions

        #region Tooltips

        private void OFTooltips()
        {
            ToolTip OFToolTips = new ToolTip();

            // Set up the delays for the ToolTip.
            OFToolTips.AutoPopDelay = 5000;
            OFToolTips.InitialDelay = 500;
            OFToolTips.ReshowDelay = 500;
            // Force the ToolTip text to be displayed whether or not the form is active.
            OFToolTips.ShowAlways = true;

            // Set up the ToolTip text for the Button and Checkbox.
            OFToolTips.SetToolTip(this.txtCaseName, "Enter a reference for the case. If left null, program will store analysis results in 'OpenForensics Output'.");
            OFToolTips.SetToolTip(this.txtEvidenceName, "Enter a unique reference for the data analysed. If left null, program will default to drive or file name.");
            OFToolTips.SetToolTip(this.btnCustom, "Modify analysis technology settings.");
            OFToolTips.SetToolTip(this.btnDefault, "Revert to recommended settings for performing analysis.");
            OFToolTips.SetToolTip(this.btnCarve, "Analyse the drive or file and identify (and optionally recreate) any files found.");
            OFToolTips.SetToolTip(this.btnAddKeyword, "Add the keyword to the keyword list.");
            OFToolTips.SetToolTip(this.btnRemoveKeyword, "Remove the currently selected keyword from the keyword list.");
            OFToolTips.SetToolTip(this.btnClearKeywords, "Clear all keywords in the keyword list.");
            OFToolTips.SetToolTip(this.btnFileOpen, "Open file to analyse.");
            OFToolTips.SetToolTip(this.btnDriveOpen, "Open physical drive to analyse.");
            OFToolTips.SetToolTip(this.chkImagePreview, "Generate Image Previews whilst processing. [High RAM Usage with large datasets!]");
        }

        #endregion

        #region Platform Selection

        private void rdoGPU_CheckedChanged(object sender, EventArgs e)
        {
            PopulateGPGPUComboBox();
        }

        private void rdoCPU_CheckedChanged(object sender, EventArgs e)
        {
            PopulateGPGPUComboBox();
        }

        #endregion

        #region Target Selection

        private void rdoFile_CheckedChanged(object sender, EventArgs e)
        {
            TargetTypeUpdate();
        }

        private void rdoKeyword_CheckedChanged(object sender, EventArgs e)
        {
            TargetTypeUpdate();
            if (cboKeywords.Items.Count == 1)
                cboKeywords.Items[0] = "No keywords present - add keywords below";
        }

        private void TargetTypeUpdate()
        {
            if (rdoFile.Checked == true)
            {
                cboFileType.Enabled = true;
                cboKeywords.Enabled = false;
                txtInput.Enabled = false;
                btnAddKeyword.Enabled = false;
                btnRemoveKeyword.Enabled = false;
                btnClearKeywords.Enabled = false;
            }
            else
            {
                cboFileType.Enabled = false;
                cboKeywords.Enabled = true;
                txtInput.Enabled = true;
                btnAddKeyword.Enabled = true;
                btnRemoveKeyword.Enabled = true;
                btnClearKeywords.Enabled = true;
            }
        }

        private void btnAddKeyword_Click(object sender, EventArgs e)
        {
            cboKeywords.Items.Add(txtInput.Text);
            cboKeywords.SelectedIndex = cboKeywords.Items.Count - 1;
            txtInput.Text = "";
            txtInput.Focus();
            if (cboKeywords.Items.Count > 1)
                cboKeywords.Items[0] = "All Keywords";
        }

        private void btnRemoveKeyword_Click(object sender, EventArgs e)
        {
            if (cboKeywords.SelectedIndex != 0)
            {
                cboKeywords.Items.RemoveAt(cboKeywords.SelectedIndex);
                cboKeywords.SelectedIndex = cboKeywords.Items.Count - 1;

                if (cboKeywords.Items.Count == 1)
                    cboKeywords.Items[0] = "No keywords present - add keywords below";
            }
        }

        private void btnClearKeywords_Click(object sender, EventArgs e)
        {
            cboKeywords.Items.Clear();
            cboKeywords.Items.Add("No keywords present - add keywords below");
            cboKeywords.SelectedIndex = cboKeywords.Items.Count - 1;
        }

        #endregion

        #region Buttons

        private void pbLogo_Click(object sender, EventArgs e)
        {
            MessageBox.Show("OpenForensics is an open-source OpenCL Digital Forensics analysis and file carving tool. This tool was built in conjunction with PhD research from Dr Ethan Bayne as a platform to demonstrate the performance enhancements possible when applying GPGPU processing and an optimised PFAC algorithm to the problem of string searching in the field of Digital Forensics. \n\nLicensed under the Apache License, version 2.0. \nFull license information can be found on GitHub: https://github.com/ethanbayne/OpenForensics", "About OpenForensics", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void btnCustom_Click(object sender, EventArgs e)
        {
            grpDefaultPlatform.Enabled = false;
            grpCustomPlatform.Enabled = true;
            grpCustomPlatform.BringToFront();
        }

        private void btnDefault_Click(object sender, EventArgs e)
        {
            rdoGPU.Checked = true;
            PopulateGPGPUComboBox();
            grpCustomPlatform.Enabled = false;
            grpDefaultPlatform.Enabled = true;
            grpDefaultPlatform.BringToFront();
        }

        private void btnAnalyse_Click(object sender, EventArgs e)
        {
            AnalysisSetup();
        }


        private void btnCarve_Click(object sender, EventArgs e)
        {
            AnalysisSetup();
        }

        private void btnDriveOpen_Click(object sender, EventArgs e)
        {
            PhysicalDriveDialog d = new PhysicalDriveDialog();
            if (d.ShowDialog() == DialogResult.OK)
            {
                txtFile.Text = d.physicalDrive;
                btnDriveOpen.BackColor = Color.DarkSeaGreen;
                btnFileOpen.BackColor = SystemColors.Control;
            }
            d.Dispose();
        }

        private void btnFileOpen_Click(object sender, EventArgs e)
        {
            // File browser for DD Selection
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                txtFile.Text = openFileDialog.InitialDirectory + openFileDialog.FileName;
                if (!FileOpenedElsewhere(openFileDialog.FileName.ToString()))
                {
                    fileName = openFileDialog.FileName.ToString();
                    btnFileOpen.BackColor = Color.DarkSeaGreen;
                    btnDriveOpen.BackColor = SystemColors.Control;
                }
                else
                {
                    MessageBox.Show("Selected file is not accessible. This file may currently be in use by another process.", "File locked!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }
        
        private void chkImagePreview_CheckedChanged(object sender, EventArgs e)
        {
            imagePreview = chkImagePreview.Checked;
        }

        #endregion

        #region Reactive

        private void OpenForensics_Resize(object sender, EventArgs e)
        {
            this.Invalidate();
        }

        private void PopulateGPGPUComboBox()
        {
            gpus.Clear();

            if (rdoCPU.Checked == true)     // If CPU is Selected, disable GPGPU combo box and set TestType to CPU
            {
                cbGPGPU.Text = "Disabled";
                cbGPGPU.Enabled = false;
                TestType = "CPU";
            }
            if (rdoGPU.Checked == true)  // If OpenCL is Selected:
            {
                cbGPGPU.Items.Clear();                          // Clear GPGPU Selection box
                multiGPU = false;

                try
                {
                    if (CudafyHost.GetDeviceCount(CudafyModes.Target = eGPUType.OpenCL) >= 1)
                    {
                        int gpuCount = 0;
                        foreach (GPGPUProperties prop in CudafyHost.GetDeviceProperties(CudafyModes.Target = eGPUType.OpenCL))
                            if (!prop.Name.Contains("CPU") && !prop.Name.Contains("Processor"))
                                gpuCount++;

                        if (gpuCount > 1)
                        {
                            cbGPGPU.Items.Add("Multi-GPU");
                            multiGPU = true;
                            cbGPGPU.SelectedIndex = 0;
                            lblPlatformDefault.Text = "Recommended Default Settings (Multi-GPU)";
                        }

                        foreach (GPGPUProperties prop in CudafyHost.GetDeviceProperties(CudafyModes.Target = eGPUType.OpenCL))
                        {
                            // Add all GPGPUs to combo box belonging to OpenCL
                            cbGPGPU.Items.Add(prop.Name.Trim() + "   ||   OpenCL platform: " + prop.PlatformName.Trim());
                            if (multiGPU == false && (!prop.Name.Contains("CPU") && !prop.Name.Contains("Processor")))
                            {
                                cbGPGPU.SelectedIndex = cbGPGPU.Items.Count - 1;
                                lblPlatformDefault.Text = "Recommended Default Settings (" + prop.Name.Trim() + ")";
                            }
                            if (multiGPU == true && (!prop.Name.Contains("CPU") && !prop.Name.Contains("Processor")))
                                gpus.Add(prop.DeviceId);

                            gpuMem.Add(prop.TotalGlobalMem);

                            if (!prop.Name.Contains("CPU") && !prop.Name.Contains("Processor"))
                                if (maxGPUMem == 0 || prop.TotalGlobalMem < maxGPUMem)
                                    maxGPUMem = prop.TotalGlobalMem;
                            //MessageBox.Show(maxGPUMem.ToString());
                        }
                        cbGPGPU.Enabled = true;     // Enable combo box
                    }
                    else
                    {
                        cbGPGPU.Items.Add("No GPU detected on system");
                        cbGPGPU.SelectedIndex = cbGPGPU.Items.Count - 1;
                        lblPlatformDefault.Text = "Recommended Default Settings (CPU)";
                        cbGPGPU.Enabled = false;
                        rdoGPU.Enabled = false;
                        rdoCPU.Checked = true;
                    }
                }
                catch
                {
                    cbGPGPU.Items.Add("Error populating GPUs on system");
                    cbGPGPU.SelectedIndex = cbGPGPU.Items.Count - 1;
                    lblPlatformDefault.Text = "Recommended Default Settings (CPU)";
                    cbGPGPU.Enabled = false;
                    rdoGPU.Enabled = false;
                    rdoCPU.Checked = true;
                }
            }
        }

        private void cbGPGPU_SelectedIndexChanged(object sender, EventArgs e)
        {
            TestType = "OpenCL";        // Set TestType to OpenCL
            CudafyModes.Target = eGPUType.OpenCL;           // Set Target to OpenCL
            CudafyTranslator.Language = eLanguage.OpenCL;   // Set Cudafy Translator Language to OpenCL (Kernel Construction)
            if (multiGPU)
                CudafyModes.DeviceId = cbGPGPU.SelectedIndex - 1;
            else
                CudafyModes.DeviceId = cbGPGPU.SelectedIndex;
        }

        private void AnalysisSetup()
        {
            if (InputCheck())   // Check that inputs are present before test
            {
                DialogResult result = folderBrowserDialog.ShowDialog();
                if (result == DialogResult.OK)
                {
                    CaseName = txtCaseName.Text;
                    EvidenceName = txtEvidenceName.Text;
                    if (CaseName == "")
                        CaseName = "OpenForensics Output";
                    if (EvidenceName == "")
                        EvidenceName = Path.GetFileName(txtFile.Text);
                    saveLocation = folderBrowserDialog.SelectedPath + "\\" + CaseName + "\\" + EvidenceName + "\\";
                    //MessageBox.Show(pathCheck);
                    if (!Directory.Exists(saveLocation))
                    {
                        Directory.CreateDirectory(saveLocation);
                        BeginAnalysis();
                    }
                    else
                    {
                        if (File.Exists(saveLocation + "CarvableFileData.of"))
                        {
                            DialogResult dialogResult = MessageBox.Show("Detected existing analysis results for " + Path.GetFileName(txtFile.Text) + "\nWould you like to reproduce the carvable files?\n\nYes:\tReproduce Files\nNo:\tReanalyse and overwrite results\nCancel:\tGo back to main menu", "Detected existing results", MessageBoxButtons.YesNoCancel, MessageBoxIcon.Warning);

                            if (dialogResult == DialogResult.Yes)
                            {
                                carvableFileRecord = "CarvableFileData.of";
                                BeginAnalysis();
                            }
                            else if (dialogResult == DialogResult.No)
                            {
                                carvableFileRecord = "";
                                while (true)
                                {
                                    try
                                    {
                                        Directory.Delete(saveLocation, true);
                                        break;
                                    }
                                    catch
                                    {
                                        DialogResult dialogConfirm2 = MessageBox.Show("Cannot overwrite " + Path.GetFileName(txtFile.Text) + "!\nRetry overwrite?", "Error Overwriting", MessageBoxButtons.YesNo, MessageBoxIcon.Error);
                                        if (dialogConfirm2 == DialogResult.Yes)
                                            Thread.Sleep(500);
                                        else
                                            return;
                                    }
                                }
                                Thread.Sleep(500);
                                Directory.CreateDirectory(saveLocation);
                                BeginAnalysis();
                            }
                            else
                            {
                                MessageBox.Show("Aborted, Returning to Main Menu", "Aborted", MessageBoxButtons.OK, MessageBoxIcon.Information);
                                return;
                            }
                        }
                        else
                        {
                            while (true)
                            {
                                try
                                {
                                    Directory.Delete(saveLocation, true);
                                    break;
                                }
                                catch
                                {
                                    DialogResult dialogConfirm2 = MessageBox.Show("Cannot overwrite " + Path.GetFileName(txtFile.Text) + "!\nRetry overwrite?", "Error Overwriting", MessageBoxButtons.YesNo, MessageBoxIcon.Error);
                                    if (dialogConfirm2 == DialogResult.Yes)
                                        Thread.Sleep(500);
                                    else
                                        return;
                                }
                            }
                            Thread.Sleep(500);
                            Directory.CreateDirectory(saveLocation);
                            BeginAnalysis();
                        }
                    }
                }
            }
        }

        private bool InputCheck()
        {
            if (txtFile.Text.StartsWith("\\\\.\\"))
            {
                ManagementObjectSearcher mosDisks = new ManagementObjectSearcher("SELECT * FROM Win32_DiskDrive WHERE DeviceID = '" + txtFile.Text.Replace("\\", "\\\\") + "'");
                if (mosDisks.Get().Count == 0)
                {
                    MessageBox.Show("Selected drive cannot be found - Check drive connection", "Cannot find selected drive", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
            }
            else
            {
                if (File.Exists(txtFile.Text) == false)
                {
                    MessageBox.Show("Selected file cannot be found - Check if file exists", "Cannot find selected file", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
                if (FileOpenedElsewhere(txtFile.Text))
                {
                    MessageBox.Show("Selected file is not accessible. This file may currently be in use by another process.", "File not accessible!", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
            }

            if (rdoGPU.Checked == true)
                if (cbGPGPU.Items.Count < 1)
                {
                    MessageBox.Show("No compatible GPU found on Computer", "No GPU Detected", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }

            if (rdoFile.Checked == true)
                if (cboFileType.Items.Count < 7)
                {
                    MessageBox.Show("No File Types present - check FileType.xml", "File Target Import Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }

            if (rdoKeyword.Checked == true)
                if (cboKeywords.Items.Count <= 1)
                {
                    MessageBox.Show("No Keywords present", "Target Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }

            if (File.Exists(txtFile.Text))
            {
                return true;
            }
            else
            {
                //MessageBox.Show("File path invalid", "File Path Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return true; //changed to true for testing
            }
        }

        #endregion

        #endregion


        #region File Handling

        private void PopulateFileTypes()
        {
            cboFileType.Items.Clear();                          // Clear File Type choices and add collections
            cboFileType.Items.Add("All File Types");
            cboFileType.Items.Add("All Image Files Types");
            cboFileType.Items.Add("All Video Files Types");
            cboFileType.Items.Add("All Audio Files Types");
            cboFileType.Items.Add("All Document Files Types");
            cboFileType.Items.Add("All Misc Files Types");

            try
            {
                XmlDocument xmldoc = new XmlDocument();             // Read File Type list from XML
                FileStream fs = new FileStream("FileTypes.xml", FileMode.Open, FileAccess.Read);
                xmldoc.Load(fs);
                XmlNodeList xmlnode = xmldoc.DocumentElement.ChildNodes;
                foreach (XmlNode childnode in xmlnode)
                {
                    string fileType = childnode["Type"].InnerText.Trim();
                    string fileName = childnode["Name"].InnerText.Trim();
                    cboFileType.Items.Add(fileName);

                    switch (fileType)
                    {
                        case "Image":
                            imageNames.Add(fileName);
                            break;
                        case "Video":
                            videoNames.Add(fileName);
                            break;
                        case "Audio":
                            audioNames.Add(fileName);
                            break;
                        case "Document":
                            documentNames.Add(fileName);
                            break;
                        case "Misc":
                            miscNames.Add(fileName);
                            break;
                        default:
                            Console.WriteLine(" [!] XML File error - please check the file type of entry: " + childnode["Name"].InnerText.Trim());
                            break;
                    }
                }
            }
            catch
            {
                MessageBox.Show("Error loading FileType.xml config file. Please make sure it exists!", "Config File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

            cboFileType.SelectedIndex = 0;  // Set File Type to first entry on list

            if (cboFileType.Items.Count == 6)
                cboFileType.Items[0] = "No filetypes present - please check FileTypes.xml config!";
        }

        private void GetFileType()
        {
            int fileTypePos = 0;
            string fileTypeValue = "";
            int keywordCount = 0;
            int keywordPos = 0;
            string keywordValue = "";
            this.Invoke((MethodInvoker)delegate ()
            {
                fileTypePos = this.cboFileType.SelectedIndex;
                fileTypeValue = this.cboFileType.SelectedItem.ToString();
                keywordCount = this.cboKeywords.Items.Count;
                keywordPos = this.cboKeywords.SelectedIndex;
                keywordValue = this.cboKeywords.SelectedItem.ToString();
            });

            if (rdoFile.Checked == true)                            // If File Type Search Selected:
            {
                switch (fileTypePos)                  // If category selected, add ranges associated
                {
                    case 0:
                        foreach (string fileType in imageNames)
                            XmlLoad(fileType);
                        foreach (string fileType in videoNames)
                            XmlLoad(fileType);
                        foreach (string fileType in audioNames)
                            XmlLoad(fileType);
                        foreach (string fileType in documentNames)
                            XmlLoad(fileType);
                        foreach (string fileType in miscNames)
                            XmlLoad(fileType);
                        break;
                    case 1:
                        foreach (string imageType in imageNames)
                            XmlLoad(imageType);
                        break;
                    case 2:
                        foreach (string fileType in videoNames)
                            XmlLoad(fileType);
                        break;
                    case 3:
                        foreach (string fileType in audioNames)
                            XmlLoad(fileType);
                        break;
                    case 4:
                        foreach (string fileType in documentNames)
                            XmlLoad(fileType);
                        break;
                    case 5:
                        foreach (string fileType in miscNames)
                            XmlLoad(fileType);
                        break;
                    default:                                        // Else, import individual values from XML
                        XmlLoad(fileTypeValue);
                        break;
                }
            }
            else                                                    // If Keyword Search Selected
            {
                if (keywordPos == 0)                 // If All Keywords selected, generate Values for Keyword List
                {
                    for (int i = 1; i < keywordCount; i++)
                    {
                        this.Invoke((MethodInvoker)delegate ()
                        {
                            keywordValue = this.cboKeywords.Items[i].ToString();
                        });
                        targetName.Add("\"" + keywordValue + "\"");
                        targetHeader.Add(Engine.StringtoHex(keywordValue));
                        targetFooter.Add(null);
                        targetLength.Add(0);
                    }
                }
                else                                                // Generate Value for Individual Selected
                {
                    targetName.Add(keywordValue);
                    targetHeader.Add(Engine.StringtoHex(keywordValue));
                    targetFooter.Add(null);
                    targetLength.Add(0);
                }
            }
        }

        private void XmlLoad(string fileType)
        {
            XmlDocument xmldoc = new XmlDocument();
            FileStream fs = new FileStream("FileTypes.xml", FileMode.Open, FileAccess.Read);
            xmldoc.Load(fs);
            XmlNodeList xmlnode = xmldoc.DocumentElement.ChildNodes;
            foreach (XmlNode childnode in xmlnode)
            {
                string typeName = childnode["Name"].InnerText.Trim();
                if (typeName == fileType)
                {
                    XmlNodeList values = childnode.SelectNodes("Value");

                    string fileEOF = null;
                    if (childnode.SelectSingleNode("EOF") != null)
                        fileEOF = childnode["EOF"].InnerText.Trim();

                    int fileLength = 10 * 1048576;
                    if (childnode.SelectSingleNode("MaxLengthMB") != null)
                        fileLength = (int)(Convert.ToDouble(childnode["MaxLengthMB"].InnerText.Trim()) * 1048576);

                    foreach (XmlNode value in values)
                    {
                        targetName.Add(typeName);
                        targetHeader.Add(value.InnerText);
                        targetFooter.Add(fileEOF);
                        targetLength.Add(fileLength);
                    }

                    break;
                }
            }
        }

        private bool HasEOF(string fileType)
        {
            XmlDocument xmldoc = new XmlDocument();
            FileStream fs = new FileStream("FileTypes.xml", FileMode.Open, FileAccess.Read);
            xmldoc.Load(fs);
            XmlNodeList xmlnode = xmldoc.DocumentElement.ChildNodes;
            foreach (XmlNode childnode in xmlnode)
            {
                string typeName = childnode["Name"].InnerText.Trim();
                if (typeName == fileType)
                {
                    if (childnode.SelectSingleNode("EOF") != null)
                        return true;
                    else
                        return false;
                }
            }

            return false;
        }

        private bool FileOpenedElsewhere(string fileLoc)
        {
            FileStream fs = null;
            try
            {
                fs = new FileStream(fileLoc, FileMode.Open, FileAccess.Read);
            }
            catch (IOException) { return true; }    // If IO Exception, return true
            finally
            {
                if (fs != null)
                    fs.Close();
            }

            return false;   // If opened correctly, return false
        }

        #endregion


        private void BeginAnalysis()
        {
            string gpuChoice = "";
            string fileType = "";
            string keyword = "";
            int filePos = 0;
            int keywordPos = 0;
            Boolean fileChecked = false;

            this.Invoke((MethodInvoker)delegate ()
            {
                if (rdoGPU.Checked)
                    gpuChoice = this.cbGPGPU.SelectedItem.ToString();
                fileType = this.cboFileType.SelectedItem.ToString();
                keyword = this.cboKeywords.SelectedItem.ToString();
                filePos = cboFileType.SelectedIndex;
                keywordPos = cboKeywords.SelectedIndex;
                fileChecked = rdoFile.Checked;
            });

            targetName.Clear();
            targetHeader.Clear();                               // Clear Search Targets
            targetFooter.Clear();

            if (rdoGPU.Checked)
                if (gpuChoice.Contains("CPU") || gpuChoice.Contains("Processor"))
                {
                    DialogResult dialogResult = MessageBox.Show("Running OpenCL on the CPU is slow.\nAre you sure you want to continue?", "GPU Selection Check", MessageBoxButtons.YesNo, MessageBoxIcon.Question);
                    if (dialogResult == DialogResult.No)
                        return;
                }

            GetFileType();                                      // Populate Search Targets


            string gpuValue = "";
            if (rdoGPU.Checked)
            {
                this.Invoke((MethodInvoker)delegate ()
                {
                    gpuValue = this.cbGPGPU.SelectedItem.ToString();
                });
                gpuValue = gpuValue.Split('|')[0].Trim();
            }

            this.Hide();
            Analysis anFrm = new Analysis();
            anFrm.Owner = this;

            Analysis.Input input = new Analysis.Input();
            input.TestType = TestType;
            input.GPGPU = gpuValue;
            if (gpuValue == "Multi-GPU")
            {
                input.gpus = gpus;
                input.maxGPUMem = maxGPUMem;
            }
            else
            {
                if (multiGPU && rdoGPU.Checked == true)
                {
                    input.gpus = new List<int>() { (this.cbGPGPU.SelectedIndex - 1) };
                    input.maxGPUMem = gpuMem[this.cbGPGPU.SelectedIndex - 1];
                }
                else
                {
                    input.gpus = new List<int>() { (this.cbGPGPU.SelectedIndex) };
                    input.maxGPUMem = gpuMem[this.cbGPGPU.SelectedIndex];
                }
            }
            input.FilePath = txtFile.Text;
            input.CaseName = CaseName;
            input.EvidenceName = EvidenceName;
            input.saveLocation = saveLocation;
            input.CarveFilePath = carvableFileRecord;
            input.targetName = targetName;
            input.targetHeader = targetHeader;
            input.targetFooter = targetFooter;
            input.targetLength = targetLength;
            input.imagePreview = imagePreview;

            anFrm.InputSet = input;
            anFrm.Show();
        }
    }
}
