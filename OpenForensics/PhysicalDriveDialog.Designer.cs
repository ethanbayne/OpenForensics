namespace OpenForensics
{
    partial class PhysicalDriveDialog
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(PhysicalDriveDialog));
            this.btnSelect = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.lblFirmware = new System.Windows.Forms.Label();
            this.lblInterface = new System.Windows.Forms.Label();
            this.lblTracksPerCyl = new System.Windows.Forms.Label();
            this.lblSectorsPerTrack = new System.Windows.Forms.Label();
            this.lblBytesPerSect = new System.Windows.Forms.Label();
            this.lblCapacity = new System.Windows.Forms.Label();
            this.lblHeads = new System.Windows.Forms.Label();
            this.lblSerial = new System.Windows.Forms.Label();
            this.lblTracks = new System.Windows.Forms.Label();
            this.lblSectors = new System.Windows.Forms.Label();
            this.lblCylinders = new System.Windows.Forms.Label();
            this.lblSignature = new System.Windows.Forms.Label();
            this.lblPartitions = new System.Windows.Forms.Label();
            this.lblSelect = new System.Windows.Forms.Label();
            this.cmbHdd = new System.Windows.Forms.ComboBox();
            this.lblModel = new System.Windows.Forms.Label();
            this.lblType = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // btnSelect
            // 
            this.btnSelect.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSelect.Location = new System.Drawing.Point(122, 303);
            this.btnSelect.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnSelect.Name = "btnSelect";
            this.btnSelect.Size = new System.Drawing.Size(87, 28);
            this.btnSelect.TabIndex = 2;
            this.btnSelect.Text = "Select";
            this.btnSelect.UseVisualStyleBackColor = true;
            this.btnSelect.Click += new System.EventHandler(this.btnSelect_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.Location = new System.Drawing.Point(217, 303);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(87, 28);
            this.btnCancel.TabIndex = 3;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.UseVisualStyleBackColor = true;
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // lblFirmware
            // 
            this.lblFirmware.AutoSize = true;
            this.lblFirmware.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblFirmware.Location = new System.Drawing.Point(12, 178);
            this.lblFirmware.Name = "lblFirmware";
            this.lblFirmware.Size = new System.Drawing.Size(59, 16);
            this.lblFirmware.TabIndex = 36;
            this.lblFirmware.Text = "Firmware:";
            // 
            // lblInterface
            // 
            this.lblInterface.AutoSize = true;
            this.lblInterface.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblInterface.Location = new System.Drawing.Point(12, 146);
            this.lblInterface.Name = "lblInterface";
            this.lblInterface.Size = new System.Drawing.Size(60, 16);
            this.lblInterface.TabIndex = 35;
            this.lblInterface.Text = "Interface:";
            // 
            // lblTracksPerCyl
            // 
            this.lblTracksPerCyl.AutoSize = true;
            this.lblTracksPerCyl.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTracksPerCyl.Location = new System.Drawing.Point(12, 277);
            this.lblTracksPerCyl.Name = "lblTracksPerCyl";
            this.lblTracksPerCyl.Size = new System.Drawing.Size(111, 16);
            this.lblTracksPerCyl.TabIndex = 34;
            this.lblTracksPerCyl.Text = "Tracks per Cylinder:";
            // 
            // lblSectorsPerTrack
            // 
            this.lblSectorsPerTrack.AutoSize = true;
            this.lblSectorsPerTrack.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSectorsPerTrack.Location = new System.Drawing.Point(12, 261);
            this.lblSectorsPerTrack.Name = "lblSectorsPerTrack";
            this.lblSectorsPerTrack.Size = new System.Drawing.Size(101, 16);
            this.lblSectorsPerTrack.TabIndex = 33;
            this.lblSectorsPerTrack.Text = "Sectors per Track:";
            // 
            // lblBytesPerSect
            // 
            this.lblBytesPerSect.AutoSize = true;
            this.lblBytesPerSect.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblBytesPerSect.Location = new System.Drawing.Point(12, 245);
            this.lblBytesPerSect.Name = "lblBytesPerSect";
            this.lblBytesPerSect.Size = new System.Drawing.Size(96, 16);
            this.lblBytesPerSect.TabIndex = 32;
            this.lblBytesPerSect.Text = "Bytes per Sector:";
            // 
            // lblCapacity
            // 
            this.lblCapacity.AutoSize = true;
            this.lblCapacity.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCapacity.Location = new System.Drawing.Point(12, 73);
            this.lblCapacity.Name = "lblCapacity";
            this.lblCapacity.Size = new System.Drawing.Size(63, 16);
            this.lblCapacity.TabIndex = 31;
            this.lblCapacity.Text = "Capacity:";
            // 
            // lblHeads
            // 
            this.lblHeads.AutoSize = true;
            this.lblHeads.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHeads.Location = new System.Drawing.Point(160, 203);
            this.lblHeads.Name = "lblHeads";
            this.lblHeads.Size = new System.Drawing.Size(46, 16);
            this.lblHeads.TabIndex = 30;
            this.lblHeads.Text = "Heads:";
            // 
            // lblSerial
            // 
            this.lblSerial.AutoSize = true;
            this.lblSerial.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSerial.Location = new System.Drawing.Point(12, 121);
            this.lblSerial.Name = "lblSerial";
            this.lblSerial.Size = new System.Drawing.Size(50, 16);
            this.lblSerial.TabIndex = 28;
            this.lblSerial.Text = "Serial #:";
            // 
            // lblTracks
            // 
            this.lblTracks.AutoSize = true;
            this.lblTracks.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTracks.Location = new System.Drawing.Point(160, 219);
            this.lblTracks.Name = "lblTracks";
            this.lblTracks.Size = new System.Drawing.Size(43, 16);
            this.lblTracks.TabIndex = 27;
            this.lblTracks.Text = "Tracks:";
            // 
            // lblSectors
            // 
            this.lblSectors.AutoSize = true;
            this.lblSectors.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSectors.Location = new System.Drawing.Point(12, 219);
            this.lblSectors.Name = "lblSectors";
            this.lblSectors.Size = new System.Drawing.Size(49, 16);
            this.lblSectors.TabIndex = 26;
            this.lblSectors.Text = "Sectors:";
            // 
            // lblCylinders
            // 
            this.lblCylinders.AutoSize = true;
            this.lblCylinders.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCylinders.Location = new System.Drawing.Point(12, 203);
            this.lblCylinders.Name = "lblCylinders";
            this.lblCylinders.Size = new System.Drawing.Size(59, 16);
            this.lblCylinders.TabIndex = 25;
            this.lblCylinders.Text = "Cylinders:";
            // 
            // lblSignature
            // 
            this.lblSignature.AutoSize = true;
            this.lblSignature.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSignature.Location = new System.Drawing.Point(12, 162);
            this.lblSignature.Name = "lblSignature";
            this.lblSignature.Size = new System.Drawing.Size(62, 16);
            this.lblSignature.TabIndex = 24;
            this.lblSignature.Text = "Signature:";
            // 
            // lblPartitions
            // 
            this.lblPartitions.AutoSize = true;
            this.lblPartitions.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblPartitions.Location = new System.Drawing.Point(12, 89);
            this.lblPartitions.Name = "lblPartitions";
            this.lblPartitions.Size = new System.Drawing.Size(59, 16);
            this.lblPartitions.TabIndex = 23;
            this.lblPartitions.Text = "Partitions:";
            // 
            // lblSelect
            // 
            this.lblSelect.AutoSize = true;
            this.lblSelect.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSelect.Location = new System.Drawing.Point(9, 9);
            this.lblSelect.Name = "lblSelect";
            this.lblSelect.Size = new System.Drawing.Size(145, 16);
            this.lblSelect.TabIndex = 21;
            this.lblSelect.Text = "Select a drive to analyse:";
            // 
            // cmbHdd
            // 
            this.cmbHdd.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbHdd.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.cmbHdd.FormattingEnabled = true;
            this.cmbHdd.Location = new System.Drawing.Point(12, 29);
            this.cmbHdd.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.cmbHdd.Name = "cmbHdd";
            this.cmbHdd.Size = new System.Drawing.Size(292, 24);
            this.cmbHdd.TabIndex = 1;
            this.cmbHdd.SelectedIndexChanged += new System.EventHandler(this.cmbHdd_SelectedIndexChanged);
            // 
            // lblModel
            // 
            this.lblModel.AutoSize = true;
            this.lblModel.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblModel.Location = new System.Drawing.Point(12, 57);
            this.lblModel.Name = "lblModel";
            this.lblModel.Size = new System.Drawing.Size(44, 15);
            this.lblModel.TabIndex = 22;
            this.lblModel.Text = "Model:";
            // 
            // lblType
            // 
            this.lblType.AutoSize = true;
            this.lblType.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblType.Location = new System.Drawing.Point(12, 105);
            this.lblType.Name = "lblType";
            this.lblType.Size = new System.Drawing.Size(37, 16);
            this.lblType.TabIndex = 29;
            this.lblType.Text = "Type:";
            // 
            // PhysicalDriveDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(316, 344);
            this.Controls.Add(this.lblFirmware);
            this.Controls.Add(this.lblInterface);
            this.Controls.Add(this.lblTracksPerCyl);
            this.Controls.Add(this.lblSectorsPerTrack);
            this.Controls.Add(this.lblBytesPerSect);
            this.Controls.Add(this.lblCapacity);
            this.Controls.Add(this.lblHeads);
            this.Controls.Add(this.lblType);
            this.Controls.Add(this.lblSerial);
            this.Controls.Add(this.lblTracks);
            this.Controls.Add(this.lblSectors);
            this.Controls.Add(this.lblCylinders);
            this.Controls.Add(this.lblSignature);
            this.Controls.Add(this.lblPartitions);
            this.Controls.Add(this.lblModel);
            this.Controls.Add(this.lblSelect);
            this.Controls.Add(this.cmbHdd);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnSelect);
            this.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.Name = "PhysicalDriveDialog";
            this.Text = "Select Drive to Use";
            this.Load += new System.EventHandler(this.PhysicalDriveDialog_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button btnSelect;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.Label lblFirmware;
        private System.Windows.Forms.Label lblInterface;
        private System.Windows.Forms.Label lblTracksPerCyl;
        private System.Windows.Forms.Label lblSectorsPerTrack;
        private System.Windows.Forms.Label lblBytesPerSect;
        private System.Windows.Forms.Label lblCapacity;
        private System.Windows.Forms.Label lblHeads;
        private System.Windows.Forms.Label lblSerial;
        private System.Windows.Forms.Label lblTracks;
        private System.Windows.Forms.Label lblSectors;
        private System.Windows.Forms.Label lblCylinders;
        private System.Windows.Forms.Label lblSignature;
        private System.Windows.Forms.Label lblPartitions;
        private System.Windows.Forms.Label lblSelect;
        private System.Windows.Forms.ComboBox cmbHdd;
        private System.Windows.Forms.Label lblModel;
        private System.Windows.Forms.Label lblType;
    }
}