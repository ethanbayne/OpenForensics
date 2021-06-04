﻿namespace OpenForensics
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
            this.lblPartitions = new System.Windows.Forms.Label();
            this.lblSelect = new System.Windows.Forms.Label();
            this.cmbHdd = new System.Windows.Forms.ComboBox();
            this.lblModel = new System.Windows.Forms.Label();
            this.lblType = new System.Windows.Forms.Label();
            this.SuspendLayout();
            // 
            // btnSelect
            // 
            this.btnSelect.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnSelect.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnSelect.Location = new System.Drawing.Point(122, 284);
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
            this.btnCancel.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.btnCancel.Font = new System.Drawing.Font("Bahnschrift SemiLight", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.Location = new System.Drawing.Point(217, 284);
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
            this.lblFirmware.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblFirmware.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblFirmware.Location = new System.Drawing.Point(12, 160);
            this.lblFirmware.Name = "lblFirmware";
            this.lblFirmware.Size = new System.Drawing.Size(61, 14);
            this.lblFirmware.TabIndex = 36;
            this.lblFirmware.Text = "Firmware:";
            // 
            // lblInterface
            // 
            this.lblInterface.AutoSize = true;
            this.lblInterface.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblInterface.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblInterface.Location = new System.Drawing.Point(12, 146);
            this.lblInterface.Name = "lblInterface";
            this.lblInterface.Size = new System.Drawing.Size(57, 14);
            this.lblInterface.TabIndex = 35;
            this.lblInterface.Text = "Interface:";
            // 
            // lblTracksPerCyl
            // 
            this.lblTracksPerCyl.AutoSize = true;
            this.lblTracksPerCyl.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblTracksPerCyl.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTracksPerCyl.Location = new System.Drawing.Point(12, 258);
            this.lblTracksPerCyl.Name = "lblTracksPerCyl";
            this.lblTracksPerCyl.Size = new System.Drawing.Size(111, 14);
            this.lblTracksPerCyl.TabIndex = 34;
            this.lblTracksPerCyl.Text = "Tracks per Cylinder:";
            // 
            // lblSectorsPerTrack
            // 
            this.lblSectorsPerTrack.AutoSize = true;
            this.lblSectorsPerTrack.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblSectorsPerTrack.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSectorsPerTrack.Location = new System.Drawing.Point(12, 242);
            this.lblSectorsPerTrack.Name = "lblSectorsPerTrack";
            this.lblSectorsPerTrack.Size = new System.Drawing.Size(104, 14);
            this.lblSectorsPerTrack.TabIndex = 33;
            this.lblSectorsPerTrack.Text = "Sectors per Track:";
            // 
            // lblBytesPerSect
            // 
            this.lblBytesPerSect.AutoSize = true;
            this.lblBytesPerSect.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblBytesPerSect.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblBytesPerSect.Location = new System.Drawing.Point(12, 226);
            this.lblBytesPerSect.Name = "lblBytesPerSect";
            this.lblBytesPerSect.Size = new System.Drawing.Size(98, 14);
            this.lblBytesPerSect.TabIndex = 32;
            this.lblBytesPerSect.Text = "Bytes per Sector:";
            // 
            // lblCapacity
            // 
            this.lblCapacity.AutoSize = true;
            this.lblCapacity.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblCapacity.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCapacity.Location = new System.Drawing.Point(12, 73);
            this.lblCapacity.Name = "lblCapacity";
            this.lblCapacity.Size = new System.Drawing.Size(54, 14);
            this.lblCapacity.TabIndex = 31;
            this.lblCapacity.Text = "Capacity:";
            // 
            // lblHeads
            // 
            this.lblHeads.AutoSize = true;
            this.lblHeads.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblHeads.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHeads.Location = new System.Drawing.Point(160, 184);
            this.lblHeads.Name = "lblHeads";
            this.lblHeads.Size = new System.Drawing.Size(42, 14);
            this.lblHeads.TabIndex = 30;
            this.lblHeads.Text = "Heads:";
            // 
            // lblSerial
            // 
            this.lblSerial.AutoSize = true;
            this.lblSerial.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblSerial.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSerial.Location = new System.Drawing.Point(12, 121);
            this.lblSerial.Name = "lblSerial";
            this.lblSerial.Size = new System.Drawing.Size(50, 14);
            this.lblSerial.TabIndex = 28;
            this.lblSerial.Text = "Serial #:";
            // 
            // lblTracks
            // 
            this.lblTracks.AutoSize = true;
            this.lblTracks.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblTracks.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTracks.Location = new System.Drawing.Point(160, 200);
            this.lblTracks.Name = "lblTracks";
            this.lblTracks.Size = new System.Drawing.Size(45, 14);
            this.lblTracks.TabIndex = 27;
            this.lblTracks.Text = "Tracks:";
            // 
            // lblSectors
            // 
            this.lblSectors.AutoSize = true;
            this.lblSectors.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblSectors.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSectors.Location = new System.Drawing.Point(12, 200);
            this.lblSectors.Name = "lblSectors";
            this.lblSectors.Size = new System.Drawing.Size(51, 14);
            this.lblSectors.TabIndex = 26;
            this.lblSectors.Text = "Sectors:";
            // 
            // lblCylinders
            // 
            this.lblCylinders.AutoSize = true;
            this.lblCylinders.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblCylinders.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblCylinders.Location = new System.Drawing.Point(12, 184);
            this.lblCylinders.Name = "lblCylinders";
            this.lblCylinders.Size = new System.Drawing.Size(59, 14);
            this.lblCylinders.TabIndex = 25;
            this.lblCylinders.Text = "Cylinders:";
            // 
            // lblPartitions
            // 
            this.lblPartitions.AutoSize = true;
            this.lblPartitions.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblPartitions.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblPartitions.Location = new System.Drawing.Point(12, 89);
            this.lblPartitions.Name = "lblPartitions";
            this.lblPartitions.Size = new System.Drawing.Size(62, 14);
            this.lblPartitions.TabIndex = 23;
            this.lblPartitions.Text = "Partitions:";
            // 
            // lblSelect
            // 
            this.lblSelect.AutoSize = true;
            this.lblSelect.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblSelect.Font = new System.Drawing.Font("Bahnschrift", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSelect.Location = new System.Drawing.Point(12, 9);
            this.lblSelect.Name = "lblSelect";
            this.lblSelect.Size = new System.Drawing.Size(192, 19);
            this.lblSelect.TabIndex = 21;
            this.lblSelect.Text = "Select a drive to analyse:";
            // 
            // cmbHdd
            // 
            this.cmbHdd.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbHdd.FlatStyle = System.Windows.Forms.FlatStyle.System;
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
            this.lblModel.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblModel.Font = new System.Drawing.Font("Bahnschrift SemiBold", 9F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblModel.Location = new System.Drawing.Point(12, 57);
            this.lblModel.Name = "lblModel";
            this.lblModel.Size = new System.Drawing.Size(41, 14);
            this.lblModel.TabIndex = 22;
            this.lblModel.Text = "Model:";
            // 
            // lblType
            // 
            this.lblType.AutoSize = true;
            this.lblType.FlatStyle = System.Windows.Forms.FlatStyle.System;
            this.lblType.Font = new System.Drawing.Font("Bahnschrift Light", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblType.Location = new System.Drawing.Point(12, 105);
            this.lblType.Name = "lblType";
            this.lblType.Size = new System.Drawing.Size(35, 14);
            this.lblType.TabIndex = 29;
            this.lblType.Text = "Type:";
            // 
            // PhysicalDriveDialog
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(7F, 16F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(316, 324);
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
            this.Controls.Add(this.lblPartitions);
            this.Controls.Add(this.lblModel);
            this.Controls.Add(this.lblSelect);
            this.Controls.Add(this.cmbHdd);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnSelect);
            this.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.MaximizeBox = false;
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
        private System.Windows.Forms.Label lblPartitions;
        private System.Windows.Forms.Label lblSelect;
        private System.Windows.Forms.ComboBox cmbHdd;
        private System.Windows.Forms.Label lblModel;
        private System.Windows.Forms.Label lblType;
    }
}