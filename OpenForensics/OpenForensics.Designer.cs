namespace OpenForensics
{
    partial class OpenForensics
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(OpenForensics));
            this.grpTargetFile = new System.Windows.Forms.GroupBox();
            this.txtInput = new System.Windows.Forms.TextBox();
            this.btnClearKeywords = new System.Windows.Forms.Button();
            this.btnRemoveKeyword = new System.Windows.Forms.Button();
            this.btnAddKeyword = new System.Windows.Forms.Button();
            this.rdoKeyword = new System.Windows.Forms.RadioButton();
            this.cboKeywords = new System.Windows.Forms.ComboBox();
            this.rdoFile = new System.Windows.Forms.RadioButton();
            this.cboFileType = new System.Windows.Forms.ComboBox();
            this.grpFilePath = new System.Windows.Forms.GroupBox();
            this.btnDriveOpen = new System.Windows.Forms.Button();
            this.btnFileOpen = new System.Windows.Forms.Button();
            this.txtFile = new System.Windows.Forms.TextBox();
            this.lblFilePath = new System.Windows.Forms.Label();
            this.grpCustomPlatform = new System.Windows.Forms.GroupBox();
            this.cbGPGPU = new System.Windows.Forms.ComboBox();
            this.btnDefault = new System.Windows.Forms.Button();
            this.rdoGPU = new System.Windows.Forms.RadioButton();
            this.rdoCPU = new System.Windows.Forms.RadioButton();
            this.openFileDialog = new System.Windows.Forms.OpenFileDialog();
            this.btnCarve = new System.Windows.Forms.Button();
            this.lblVersion = new System.Windows.Forms.Label();
            this.folderBrowserDialog = new System.Windows.Forms.FolderBrowserDialog();
            this.pnlMainInterface = new System.Windows.Forms.Panel();
            this.chkImagePreview = new System.Windows.Forms.CheckBox();
            this.txtEvidenceName = new System.Windows.Forms.TextBox();
            this.lblEvidenceName = new System.Windows.Forms.Label();
            this.grpDefaultPlatform = new System.Windows.Forms.GroupBox();
            this.btnCustom = new System.Windows.Forms.Button();
            this.lblPlatformDefault = new System.Windows.Forms.Label();
            this.txtCaseName = new System.Windows.Forms.TextBox();
            this.lblCaseName = new System.Windows.Forms.Label();
            this.pbLogo = new System.Windows.Forms.PictureBox();
            this.chkSkin = new System.Windows.Forms.CheckBox();
            this.grpTargetFile.SuspendLayout();
            this.grpFilePath.SuspendLayout();
            this.grpCustomPlatform.SuspendLayout();
            this.pnlMainInterface.SuspendLayout();
            this.grpDefaultPlatform.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbLogo)).BeginInit();
            this.SuspendLayout();
            // 
            // grpTargetFile
            // 
            this.grpTargetFile.Controls.Add(this.txtInput);
            this.grpTargetFile.Controls.Add(this.btnClearKeywords);
            this.grpTargetFile.Controls.Add(this.btnRemoveKeyword);
            this.grpTargetFile.Controls.Add(this.btnAddKeyword);
            this.grpTargetFile.Controls.Add(this.rdoKeyword);
            this.grpTargetFile.Controls.Add(this.cboKeywords);
            this.grpTargetFile.Controls.Add(this.rdoFile);
            this.grpTargetFile.Controls.Add(this.cboFileType);
            this.grpTargetFile.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.grpTargetFile.Font = new System.Drawing.Font("Bahnschrift", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpTargetFile.Location = new System.Drawing.Point(13, 258);
            this.grpTargetFile.Name = "grpTargetFile";
            this.grpTargetFile.Size = new System.Drawing.Size(485, 114);
            this.grpTargetFile.TabIndex = 36;
            this.grpTargetFile.TabStop = false;
            this.grpTargetFile.Text = "Select Search Target";
            // 
            // txtInput
            // 
            this.txtInput.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtInput.Location = new System.Drawing.Point(7, 84);
            this.txtInput.Name = "txtInput";
            this.txtInput.Size = new System.Drawing.Size(150, 23);
            this.txtInput.TabIndex = 9;
            // 
            // btnClearKeywords
            // 
            this.btnClearKeywords.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnClearKeywords.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnClearKeywords.Location = new System.Drawing.Point(377, 84);
            this.btnClearKeywords.Name = "btnClearKeywords";
            this.btnClearKeywords.Size = new System.Drawing.Size(101, 23);
            this.btnClearKeywords.TabIndex = 12;
            this.btnClearKeywords.Text = "Clear Keywords";
            this.btnClearKeywords.UseVisualStyleBackColor = true;
            this.btnClearKeywords.Click += new System.EventHandler(this.btnClearKeywords_Click);
            // 
            // btnRemoveKeyword
            // 
            this.btnRemoveKeyword.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnRemoveKeyword.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnRemoveKeyword.Location = new System.Drawing.Point(269, 84);
            this.btnRemoveKeyword.Name = "btnRemoveKeyword";
            this.btnRemoveKeyword.Size = new System.Drawing.Size(102, 23);
            this.btnRemoveKeyword.TabIndex = 11;
            this.btnRemoveKeyword.Text = "Delete Keyword";
            this.btnRemoveKeyword.UseVisualStyleBackColor = true;
            this.btnRemoveKeyword.Click += new System.EventHandler(this.btnRemoveKeyword_Click);
            // 
            // btnAddKeyword
            // 
            this.btnAddKeyword.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnAddKeyword.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnAddKeyword.Location = new System.Drawing.Point(161, 84);
            this.btnAddKeyword.Name = "btnAddKeyword";
            this.btnAddKeyword.Size = new System.Drawing.Size(102, 23);
            this.btnAddKeyword.TabIndex = 10;
            this.btnAddKeyword.Text = "Add Keyword";
            this.btnAddKeyword.UseVisualStyleBackColor = true;
            this.btnAddKeyword.Click += new System.EventHandler(this.btnAddKeyword_Click);
            // 
            // rdoKeyword
            // 
            this.rdoKeyword.AutoSize = true;
            this.rdoKeyword.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.rdoKeyword.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.rdoKeyword.Location = new System.Drawing.Point(15, 55);
            this.rdoKeyword.Name = "rdoKeyword";
            this.rdoKeyword.Size = new System.Drawing.Size(83, 21);
            this.rdoKeyword.TabIndex = 7;
            this.rdoKeyword.TabStop = true;
            this.rdoKeyword.Text = "Keyword";
            this.rdoKeyword.UseVisualStyleBackColor = true;
            this.rdoKeyword.CheckedChanged += new System.EventHandler(this.rdoKeyword_CheckedChanged);
            // 
            // cboKeywords
            // 
            this.cboKeywords.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboKeywords.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cboKeywords.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cboKeywords.FormattingEnabled = true;
            this.cboKeywords.Items.AddRange(new object[] {
            "No keywords present - add keywords below"});
            this.cboKeywords.Location = new System.Drawing.Point(109, 54);
            this.cboKeywords.Name = "cboKeywords";
            this.cboKeywords.Size = new System.Drawing.Size(369, 24);
            this.cboKeywords.TabIndex = 8;
            // 
            // rdoFile
            // 
            this.rdoFile.AutoSize = true;
            this.rdoFile.Checked = true;
            this.rdoFile.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.rdoFile.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.rdoFile.Location = new System.Drawing.Point(15, 28);
            this.rdoFile.Name = "rdoFile";
            this.rdoFile.Size = new System.Drawing.Size(82, 21);
            this.rdoFile.TabIndex = 5;
            this.rdoFile.TabStop = true;
            this.rdoFile.Text = "File Type";
            this.rdoFile.UseVisualStyleBackColor = true;
            this.rdoFile.CheckedChanged += new System.EventHandler(this.rdoFile_CheckedChanged);
            // 
            // cboFileType
            // 
            this.cboFileType.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cboFileType.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cboFileType.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cboFileType.FormattingEnabled = true;
            this.cboFileType.Location = new System.Drawing.Point(109, 27);
            this.cboFileType.Name = "cboFileType";
            this.cboFileType.Size = new System.Drawing.Size(369, 24);
            this.cboFileType.TabIndex = 6;
            // 
            // grpFilePath
            // 
            this.grpFilePath.Controls.Add(this.btnDriveOpen);
            this.grpFilePath.Controls.Add(this.btnFileOpen);
            this.grpFilePath.Controls.Add(this.txtFile);
            this.grpFilePath.Controls.Add(this.lblFilePath);
            this.grpFilePath.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.grpFilePath.Font = new System.Drawing.Font("Bahnschrift", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpFilePath.Location = new System.Drawing.Point(13, 60);
            this.grpFilePath.Name = "grpFilePath";
            this.grpFilePath.Size = new System.Drawing.Size(485, 192);
            this.grpFilePath.TabIndex = 35;
            this.grpFilePath.TabStop = false;
            this.grpFilePath.Text = "Select Drive or File to Analyse";
            // 
            // btnDriveOpen
            // 
            this.btnDriveOpen.BackgroundImage = global::OpenForensics.Properties.Resources.driveIcon;
            this.btnDriveOpen.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.btnDriveOpen.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnDriveOpen.Font = new System.Drawing.Font("Bahnschrift", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnDriveOpen.Location = new System.Drawing.Point(9, 28);
            this.btnDriveOpen.Name = "btnDriveOpen";
            this.btnDriveOpen.Size = new System.Drawing.Size(232, 130);
            this.btnDriveOpen.TabIndex = 3;
            this.btnDriveOpen.TextAlign = System.Drawing.ContentAlignment.BottomRight;
            this.btnDriveOpen.UseVisualStyleBackColor = true;
            this.btnDriveOpen.Click += new System.EventHandler(this.btnDriveOpen_Click);
            // 
            // btnFileOpen
            // 
            this.btnFileOpen.BackgroundImage = global::OpenForensics.Properties.Resources.fileIcon;
            this.btnFileOpen.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Zoom;
            this.btnFileOpen.FlatStyle = System.Windows.Forms.FlatStyle.Flat;
            this.btnFileOpen.Font = new System.Drawing.Font("Bahnschrift", 7.875F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnFileOpen.Location = new System.Drawing.Point(247, 28);
            this.btnFileOpen.Name = "btnFileOpen";
            this.btnFileOpen.Size = new System.Drawing.Size(232, 130);
            this.btnFileOpen.TabIndex = 4;
            this.btnFileOpen.TextAlign = System.Drawing.ContentAlignment.BottomRight;
            this.btnFileOpen.UseVisualStyleBackColor = true;
            this.btnFileOpen.Click += new System.EventHandler(this.btnFileOpen_Click);
            // 
            // txtFile
            // 
            this.txtFile.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtFile.Location = new System.Drawing.Point(43, 164);
            this.txtFile.Name = "txtFile";
            this.txtFile.ReadOnly = true;
            this.txtFile.Size = new System.Drawing.Size(436, 23);
            this.txtFile.TabIndex = 3;
            this.txtFile.TabStop = false;
            // 
            // lblFilePath
            // 
            this.lblFilePath.AutoSize = true;
            this.lblFilePath.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblFilePath.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.lblFilePath.Location = new System.Drawing.Point(12, 167);
            this.lblFilePath.Name = "lblFilePath";
            this.lblFilePath.Size = new System.Drawing.Size(34, 16);
            this.lblFilePath.TabIndex = 3;
            this.lblFilePath.Text = "Path";
            // 
            // grpCustomPlatform
            // 
            this.grpCustomPlatform.Controls.Add(this.cbGPGPU);
            this.grpCustomPlatform.Controls.Add(this.btnDefault);
            this.grpCustomPlatform.Controls.Add(this.rdoGPU);
            this.grpCustomPlatform.Controls.Add(this.rdoCPU);
            this.grpCustomPlatform.Enabled = false;
            this.grpCustomPlatform.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.grpCustomPlatform.Font = new System.Drawing.Font("Bahnschrift", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpCustomPlatform.Location = new System.Drawing.Point(13, 376);
            this.grpCustomPlatform.Name = "grpCustomPlatform";
            this.grpCustomPlatform.Size = new System.Drawing.Size(485, 52);
            this.grpCustomPlatform.TabIndex = 30;
            this.grpCustomPlatform.TabStop = false;
            this.grpCustomPlatform.Text = "Hardware Platform";
            // 
            // cbGPGPU
            // 
            this.cbGPGPU.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cbGPGPU.DropDownWidth = 450;
            this.cbGPGPU.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.cbGPGPU.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cbGPGPU.FormattingEnabled = true;
            this.cbGPGPU.Location = new System.Drawing.Point(111, 22);
            this.cbGPGPU.Name = "cbGPGPU";
            this.cbGPGPU.Size = new System.Drawing.Size(337, 24);
            this.cbGPGPU.TabIndex = 15;
            this.cbGPGPU.SelectedIndexChanged += new System.EventHandler(this.cbGPGPU_SelectedIndexChanged);
            // 
            // btnDefault
            // 
            this.btnDefault.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("btnDefault.BackgroundImage")));
            this.btnDefault.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.btnDefault.Font = new System.Drawing.Font("Bahnschrift", 6.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnDefault.Location = new System.Drawing.Point(454, 24);
            this.btnDefault.Name = "btnDefault";
            this.btnDefault.Size = new System.Drawing.Size(25, 23);
            this.btnDefault.TabIndex = 17;
            this.btnDefault.UseVisualStyleBackColor = true;
            this.btnDefault.Click += new System.EventHandler(this.btnDefault_Click);
            // 
            // rdoGPU
            // 
            this.rdoGPU.AutoSize = true;
            this.rdoGPU.Checked = true;
            this.rdoGPU.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.rdoGPU.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.rdoGPU.Location = new System.Drawing.Point(60, 25);
            this.rdoGPU.Name = "rdoGPU";
            this.rdoGPU.Size = new System.Drawing.Size(57, 21);
            this.rdoGPU.TabIndex = 12;
            this.rdoGPU.TabStop = true;
            this.rdoGPU.Text = "GPU";
            this.rdoGPU.UseVisualStyleBackColor = true;
            this.rdoGPU.CheckedChanged += new System.EventHandler(this.rdoGPU_CheckedChanged);
            // 
            // rdoCPU
            // 
            this.rdoCPU.AutoSize = true;
            this.rdoCPU.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.rdoCPU.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.rdoCPU.Location = new System.Drawing.Point(10, 25);
            this.rdoCPU.Name = "rdoCPU";
            this.rdoCPU.Size = new System.Drawing.Size(57, 21);
            this.rdoCPU.TabIndex = 11;
            this.rdoCPU.TabStop = true;
            this.rdoCPU.Text = "CPU";
            this.rdoCPU.UseVisualStyleBackColor = true;
            this.rdoCPU.CheckedChanged += new System.EventHandler(this.rdoCPU_CheckedChanged);
            // 
            // openFileDialog
            // 
            this.openFileDialog.Filter = "DD files|*.dd|All files|*.*";
            // 
            // btnCarve
            // 
            this.btnCarve.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnCarve.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCarve.Location = new System.Drawing.Point(305, 436);
            this.btnCarve.Name = "btnCarve";
            this.btnCarve.Size = new System.Drawing.Size(193, 25);
            this.btnCarve.TabIndex = 18;
            this.btnCarve.Text = "Begin Analysis";
            this.btnCarve.UseVisualStyleBackColor = true;
            this.btnCarve.Click += new System.EventHandler(this.btnCarve_Click);
            // 
            // lblVersion
            // 
            this.lblVersion.BackColor = System.Drawing.Color.Transparent;
            this.lblVersion.Font = new System.Drawing.Font("Bahnschrift SemiLight", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblVersion.Location = new System.Drawing.Point(358, 35);
            this.lblVersion.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lblVersion.Name = "lblVersion";
            this.lblVersion.Size = new System.Drawing.Size(168, 32);
            this.lblVersion.TabIndex = 42;
            this.lblVersion.Text = "Version Info v. x.x";
            this.lblVersion.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // folderBrowserDialog
            // 
            this.folderBrowserDialog.Description = "Save Location";
            // 
            // pnlMainInterface
            // 
            this.pnlMainInterface.BackColor = System.Drawing.SystemColors.Control;
            this.pnlMainInterface.BorderStyle = System.Windows.Forms.BorderStyle.FixedSingle;
            this.pnlMainInterface.Controls.Add(this.chkSkin);
            this.pnlMainInterface.Controls.Add(this.chkImagePreview);
            this.pnlMainInterface.Controls.Add(this.txtEvidenceName);
            this.pnlMainInterface.Controls.Add(this.lblEvidenceName);
            this.pnlMainInterface.Controls.Add(this.grpDefaultPlatform);
            this.pnlMainInterface.Controls.Add(this.txtCaseName);
            this.pnlMainInterface.Controls.Add(this.lblCaseName);
            this.pnlMainInterface.Controls.Add(this.grpCustomPlatform);
            this.pnlMainInterface.Controls.Add(this.grpFilePath);
            this.pnlMainInterface.Controls.Add(this.btnCarve);
            this.pnlMainInterface.Controls.Add(this.grpTargetFile);
            this.pnlMainInterface.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.pnlMainInterface.Location = new System.Drawing.Point(13, 67);
            this.pnlMainInterface.Name = "pnlMainInterface";
            this.pnlMainInterface.Size = new System.Drawing.Size(512, 473);
            this.pnlMainInterface.TabIndex = 43;
            // 
            // chkImagePreview
            // 
            this.chkImagePreview.AutoSize = true;
            this.chkImagePreview.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.chkImagePreview.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.chkImagePreview.Location = new System.Drawing.Point(13, 439);
            this.chkImagePreview.Name = "chkImagePreview";
            this.chkImagePreview.Size = new System.Drawing.Size(147, 21);
            this.chkImagePreview.TabIndex = 17;
            this.chkImagePreview.Text = "Live Image Preview";
            this.chkImagePreview.UseVisualStyleBackColor = true;
            this.chkImagePreview.CheckedChanged += new System.EventHandler(this.chkImagePreview_CheckedChanged);
            // 
            // txtEvidenceName
            // 
            this.txtEvidenceName.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtEvidenceName.Location = new System.Drawing.Point(155, 32);
            this.txtEvidenceName.Name = "txtEvidenceName";
            this.txtEvidenceName.Size = new System.Drawing.Size(343, 23);
            this.txtEvidenceName.TabIndex = 2;
            // 
            // lblEvidenceName
            // 
            this.lblEvidenceName.AutoSize = true;
            this.lblEvidenceName.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblEvidenceName.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.lblEvidenceName.Location = new System.Drawing.Point(17, 35);
            this.lblEvidenceName.Name = "lblEvidenceName";
            this.lblEvidenceName.Size = new System.Drawing.Size(125, 16);
            this.lblEvidenceName.TabIndex = 43;
            this.lblEvidenceName.Text = "Evidence Reference:";
            // 
            // grpDefaultPlatform
            // 
            this.grpDefaultPlatform.Controls.Add(this.btnCustom);
            this.grpDefaultPlatform.Controls.Add(this.lblPlatformDefault);
            this.grpDefaultPlatform.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.grpDefaultPlatform.Font = new System.Drawing.Font("Bahnschrift", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpDefaultPlatform.Location = new System.Drawing.Point(13, 376);
            this.grpDefaultPlatform.Name = "grpDefaultPlatform";
            this.grpDefaultPlatform.Size = new System.Drawing.Size(485, 52);
            this.grpDefaultPlatform.TabIndex = 31;
            this.grpDefaultPlatform.TabStop = false;
            this.grpDefaultPlatform.Text = "Hardware Platform";
            // 
            // btnCustom
            // 
            this.btnCustom.BackgroundImage = ((System.Drawing.Image)(resources.GetObject("btnCustom.BackgroundImage")));
            this.btnCustom.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.btnCustom.Font = new System.Drawing.Font("Bahnschrift", 6.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCustom.Location = new System.Drawing.Point(454, 24);
            this.btnCustom.Name = "btnCustom";
            this.btnCustom.Size = new System.Drawing.Size(25, 23);
            this.btnCustom.TabIndex = 16;
            this.btnCustom.UseVisualStyleBackColor = true;
            this.btnCustom.Click += new System.EventHandler(this.btnCustom_Click);
            // 
            // lblPlatformDefault
            // 
            this.lblPlatformDefault.AutoSize = true;
            this.lblPlatformDefault.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblPlatformDefault.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.lblPlatformDefault.Location = new System.Drawing.Point(6, 26);
            this.lblPlatformDefault.Margin = new System.Windows.Forms.Padding(2, 0, 2, 0);
            this.lblPlatformDefault.Name = "lblPlatformDefault";
            this.lblPlatformDefault.Size = new System.Drawing.Size(196, 16);
            this.lblPlatformDefault.TabIndex = 0;
            this.lblPlatformDefault.Text = "Default Settings (Recommended)";
            // 
            // txtCaseName
            // 
            this.txtCaseName.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtCaseName.Location = new System.Drawing.Point(155, 8);
            this.txtCaseName.Name = "txtCaseName";
            this.txtCaseName.Size = new System.Drawing.Size(343, 23);
            this.txtCaseName.TabIndex = 1;
            // 
            // lblCaseName
            // 
            this.lblCaseName.AutoSize = true;
            this.lblCaseName.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblCaseName.Font = new System.Drawing.Font("Bahnschrift Light", 9.75F);
            this.lblCaseName.Location = new System.Drawing.Point(17, 11);
            this.lblCaseName.Name = "lblCaseName";
            this.lblCaseName.Size = new System.Drawing.Size(102, 16);
            this.lblCaseName.TabIndex = 41;
            this.lblCaseName.Text = "Case Reference:";
            // 
            // pbLogo
            // 
            this.pbLogo.BackColor = System.Drawing.Color.Transparent;
            this.pbLogo.BackgroundImage = global::OpenForensics.Properties.Resources.OpenForensicsLogo2;
            this.pbLogo.BackgroundImageLayout = System.Windows.Forms.ImageLayout.Stretch;
            this.pbLogo.Location = new System.Drawing.Point(-16, 12);
            this.pbLogo.Name = "pbLogo";
            this.pbLogo.Size = new System.Drawing.Size(340, 49);
            this.pbLogo.TabIndex = 44;
            this.pbLogo.TabStop = false;
            this.pbLogo.Click += new System.EventHandler(this.pbLogo_Click);
            // 
            // chkSkin
            // 
            this.chkSkin.AutoSize = true;
            this.chkSkin.Checked = true;
            this.chkSkin.CheckState = System.Windows.Forms.CheckState.Checked;
            this.chkSkin.Enabled = false;
            this.chkSkin.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.chkSkin.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.chkSkin.Location = new System.Drawing.Point(166, 439);
            this.chkSkin.Name = "chkSkin";
            this.chkSkin.Size = new System.Drawing.Size(115, 21);
            this.chkSkin.TabIndex = 44;
            this.chkSkin.Text = "Skin Detection";
            this.chkSkin.UseVisualStyleBackColor = true;
            this.chkSkin.CheckedChanged += new System.EventHandler(this.chkSkin_CheckedChanged);
            // 
            // OpenForensics
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(96F, 96F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.ClientSize = new System.Drawing.Size(537, 552);
            this.Controls.Add(this.pbLogo);
            this.Controls.Add(this.pnlMainInterface);
            this.Controls.Add(this.lblVersion);
            this.Font = new System.Drawing.Font("Bahnschrift Light", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.MaximizeBox = false;
            this.Name = "OpenForensics";
            this.Text = "OpenForensics";
            this.Load += new System.EventHandler(this.OpenForensics_Load);
            this.Resize += new System.EventHandler(this.OpenForensics_Resize);
            this.grpTargetFile.ResumeLayout(false);
            this.grpTargetFile.PerformLayout();
            this.grpFilePath.ResumeLayout(false);
            this.grpFilePath.PerformLayout();
            this.grpCustomPlatform.ResumeLayout(false);
            this.grpCustomPlatform.PerformLayout();
            this.pnlMainInterface.ResumeLayout(false);
            this.pnlMainInterface.PerformLayout();
            this.grpDefaultPlatform.ResumeLayout(false);
            this.grpDefaultPlatform.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)(this.pbLogo)).EndInit();
            this.ResumeLayout(false);

        }

        #endregion

        private System.Windows.Forms.GroupBox grpTargetFile;
        private System.Windows.Forms.TextBox txtInput;
        private System.Windows.Forms.Button btnClearKeywords;
        private System.Windows.Forms.Button btnRemoveKeyword;
        private System.Windows.Forms.Button btnAddKeyword;
        private System.Windows.Forms.RadioButton rdoKeyword;
        private System.Windows.Forms.ComboBox cboKeywords;
        private System.Windows.Forms.RadioButton rdoFile;
        private System.Windows.Forms.ComboBox cboFileType;
        private System.Windows.Forms.GroupBox grpFilePath;
        private System.Windows.Forms.Button btnFileOpen;
        private System.Windows.Forms.TextBox txtFile;
        private System.Windows.Forms.Label lblFilePath;
        private System.Windows.Forms.GroupBox grpCustomPlatform;
        private System.Windows.Forms.RadioButton rdoGPU;
        private System.Windows.Forms.ComboBox cbGPGPU;
        private System.Windows.Forms.RadioButton rdoCPU;
        private System.Windows.Forms.OpenFileDialog openFileDialog;
        private System.Windows.Forms.Button btnCarve;
        private System.Windows.Forms.Label lblVersion;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog;
        private System.Windows.Forms.Panel pnlMainInterface;
        private System.Windows.Forms.GroupBox grpDefaultPlatform;
        private System.Windows.Forms.Button btnCustom;
        private System.Windows.Forms.Label lblPlatformDefault;
        private System.Windows.Forms.Button btnDefault;
        private System.Windows.Forms.Button btnDriveOpen;
        private System.Windows.Forms.PictureBox pbLogo;
        private System.Windows.Forms.TextBox txtCaseName;
        private System.Windows.Forms.Label lblCaseName;
        private System.Windows.Forms.TextBox txtEvidenceName;
        private System.Windows.Forms.Label lblEvidenceName;
        private System.Windows.Forms.CheckBox chkImagePreview;
        private System.Windows.Forms.CheckBox chkSkin;
    }
}

