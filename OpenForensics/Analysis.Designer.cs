namespace OpenForensics
{
    partial class Analysis
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
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(Analysis));
            this.pbProgress = new System.Windows.Forms.ProgressBar();
            this.lblHeader = new System.Windows.Forms.Label();
            this.lblProgress = new System.Windows.Forms.Label();
            this.lblProcess = new System.Windows.Forms.Label();
            this.grpGPUActivity = new System.Windows.Forms.GroupBox();
            this.tblGPU = new System.Windows.Forms.TableLayoutPanel();
            this.grpProcessed = new System.Windows.Forms.GroupBox();
            this.lblFoundValue = new System.Windows.Forms.Label();
            this.lblSegmentsValue = new System.Windows.Forms.Label();
            this.lblTimeRemainingValue = new System.Windows.Forms.Label();
            this.lblTimeElapsedValue = new System.Windows.Forms.Label();
            this.lblTimeRemaining = new System.Windows.Forms.Label();
            this.lblTimeElapsed = new System.Windows.Forms.Label();
            this.lblFound = new System.Windows.Forms.Label();
            this.lblSegments = new System.Windows.Forms.Label();
            this.lstThumbs = new System.Windows.Forms.ListView();
            this.btnStop = new System.Windows.Forms.Button();
            this.btnCarve = new System.Windows.Forms.Button();
            this.grpGPUActivity.SuspendLayout();
            this.grpProcessed.SuspendLayout();
            this.SuspendLayout();
            // 
            // pbProgress
            // 
            this.pbProgress.Location = new System.Drawing.Point(12, 384);
            this.pbProgress.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.pbProgress.Name = "pbProgress";
            this.pbProgress.Size = new System.Drawing.Size(551, 26);
            this.pbProgress.Style = System.Windows.Forms.ProgressBarStyle.Continuous;
            this.pbProgress.TabIndex = 0;
            // 
            // lblHeader
            // 
            this.lblHeader.AutoSize = true;
            this.lblHeader.Font = new System.Drawing.Font("Century Gothic", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblHeader.Location = new System.Drawing.Point(12, 358);
            this.lblHeader.Name = "lblHeader";
            this.lblHeader.Size = new System.Drawing.Size(172, 22);
            this.lblHeader.TabIndex = 1;
            this.lblHeader.Text = "Carving Started...";
            // 
            // lblProgress
            // 
            this.lblProgress.Anchor = ((System.Windows.Forms.AnchorStyles)((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Right)));
            this.lblProgress.BackColor = System.Drawing.SystemColors.Control;
            this.lblProgress.Font = new System.Drawing.Font("Century Gothic", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblProgress.Location = new System.Drawing.Point(505, 414);
            this.lblProgress.Name = "lblProgress";
            this.lblProgress.Size = new System.Drawing.Size(57, 20);
            this.lblProgress.TabIndex = 2;
            this.lblProgress.Text = "0%";
            this.lblProgress.TextAlign = System.Drawing.ContentAlignment.TopRight;
            // 
            // lblProcess
            // 
            this.lblProcess.AutoSize = true;
            this.lblProcess.Font = new System.Drawing.Font("Century Gothic", 11.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblProcess.Location = new System.Drawing.Point(12, 414);
            this.lblProcess.Name = "lblProcess";
            this.lblProcess.Size = new System.Drawing.Size(91, 20);
            this.lblProcess.TabIndex = 3;
            this.lblProcess.Text = "Processing:";
            // 
            // grpGPUActivity
            // 
            this.grpGPUActivity.Controls.Add(this.tblGPU);
            this.grpGPUActivity.Font = new System.Drawing.Font("Century Gothic", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpGPUActivity.Location = new System.Drawing.Point(175, 438);
            this.grpGPUActivity.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.grpGPUActivity.Name = "grpGPUActivity";
            this.grpGPUActivity.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.grpGPUActivity.Size = new System.Drawing.Size(388, 95);
            this.grpGPUActivity.TabIndex = 4;
            this.grpGPUActivity.TabStop = false;
            this.grpGPUActivity.Text = "Thread Activity";
            // 
            // tblGPU
            // 
            this.tblGPU.AutoSize = true;
            this.tblGPU.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.tblGPU.ColumnCount = 1;
            this.tblGPU.ColumnStyles.Add(new System.Windows.Forms.ColumnStyle(System.Windows.Forms.SizeType.Absolute, 380F));
            this.tblGPU.Font = new System.Drawing.Font("Century Gothic", 6.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.tblGPU.Location = new System.Drawing.Point(3, 26);
            this.tblGPU.Margin = new System.Windows.Forms.Padding(0);
            this.tblGPU.MaximumSize = new System.Drawing.Size(380, 65);
            this.tblGPU.MinimumSize = new System.Drawing.Size(380, 65);
            this.tblGPU.Name = "tblGPU";
            this.tblGPU.RightToLeft = System.Windows.Forms.RightToLeft.No;
            this.tblGPU.RowCount = 1;
            this.tblGPU.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Percent, 100F));
            this.tblGPU.RowStyles.Add(new System.Windows.Forms.RowStyle(System.Windows.Forms.SizeType.Absolute, 65F));
            this.tblGPU.Size = new System.Drawing.Size(380, 65);
            this.tblGPU.TabIndex = 0;
            // 
            // grpProcessed
            // 
            this.grpProcessed.Controls.Add(this.lblFoundValue);
            this.grpProcessed.Controls.Add(this.lblSegmentsValue);
            this.grpProcessed.Controls.Add(this.lblTimeRemainingValue);
            this.grpProcessed.Controls.Add(this.lblTimeElapsedValue);
            this.grpProcessed.Controls.Add(this.lblTimeRemaining);
            this.grpProcessed.Controls.Add(this.lblTimeElapsed);
            this.grpProcessed.Controls.Add(this.lblFound);
            this.grpProcessed.Controls.Add(this.lblSegments);
            this.grpProcessed.Font = new System.Drawing.Font("Century Gothic", 11.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.grpProcessed.Location = new System.Drawing.Point(12, 438);
            this.grpProcessed.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.grpProcessed.Name = "grpProcessed";
            this.grpProcessed.Padding = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.grpProcessed.Size = new System.Drawing.Size(157, 95);
            this.grpProcessed.TabIndex = 6;
            this.grpProcessed.TabStop = false;
            this.grpProcessed.Text = "Status";
            // 
            // lblFoundValue
            // 
            this.lblFoundValue.AutoSize = true;
            this.lblFoundValue.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblFoundValue.Location = new System.Drawing.Point(101, 71);
            this.lblFoundValue.Name = "lblFoundValue";
            this.lblFoundValue.Size = new System.Drawing.Size(14, 16);
            this.lblFoundValue.TabIndex = 7;
            this.lblFoundValue.Text = "0";
            // 
            // lblSegmentsValue
            // 
            this.lblSegmentsValue.AutoSize = true;
            this.lblSegmentsValue.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSegmentsValue.Location = new System.Drawing.Point(101, 56);
            this.lblSegmentsValue.Name = "lblSegmentsValue";
            this.lblSegmentsValue.Size = new System.Drawing.Size(14, 16);
            this.lblSegmentsValue.TabIndex = 6;
            this.lblSegmentsValue.Text = "0";
            // 
            // lblTimeRemainingValue
            // 
            this.lblTimeRemainingValue.AutoSize = true;
            this.lblTimeRemainingValue.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTimeRemainingValue.Location = new System.Drawing.Point(101, 40);
            this.lblTimeRemainingValue.Name = "lblTimeRemainingValue";
            this.lblTimeRemainingValue.Size = new System.Drawing.Size(50, 16);
            this.lblTimeRemainingValue.TabIndex = 5;
            this.lblTimeRemainingValue.Text = "00:00:00";
            // 
            // lblTimeElapsedValue
            // 
            this.lblTimeElapsedValue.AutoSize = true;
            this.lblTimeElapsedValue.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTimeElapsedValue.Location = new System.Drawing.Point(101, 23);
            this.lblTimeElapsedValue.Name = "lblTimeElapsedValue";
            this.lblTimeElapsedValue.Size = new System.Drawing.Size(50, 16);
            this.lblTimeElapsedValue.TabIndex = 4;
            this.lblTimeElapsedValue.Text = "00:00:00";
            // 
            // lblTimeRemaining
            // 
            this.lblTimeRemaining.AutoSize = true;
            this.lblTimeRemaining.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTimeRemaining.Location = new System.Drawing.Point(6, 40);
            this.lblTimeRemaining.Name = "lblTimeRemaining";
            this.lblTimeRemaining.Size = new System.Drawing.Size(95, 16);
            this.lblTimeRemaining.TabIndex = 3;
            this.lblTimeRemaining.Text = "Time Remaining:";
            // 
            // lblTimeElapsed
            // 
            this.lblTimeElapsed.AutoSize = true;
            this.lblTimeElapsed.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblTimeElapsed.Location = new System.Drawing.Point(6, 23);
            this.lblTimeElapsed.Name = "lblTimeElapsed";
            this.lblTimeElapsed.Size = new System.Drawing.Size(81, 16);
            this.lblTimeElapsed.TabIndex = 2;
            this.lblTimeElapsed.Text = "Time Elapsed:";
            // 
            // lblFound
            // 
            this.lblFound.AutoSize = true;
            this.lblFound.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblFound.Location = new System.Drawing.Point(6, 71);
            this.lblFound.Name = "lblFound";
            this.lblFound.Size = new System.Drawing.Size(100, 16);
            this.lblFound.TabIndex = 1;
            this.lblFound.Text = "Traces Detected:";
            // 
            // lblSegments
            // 
            this.lblSegments.AutoSize = true;
            this.lblSegments.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.lblSegments.Location = new System.Drawing.Point(6, 56);
            this.lblSegments.Name = "lblSegments";
            this.lblSegments.Size = new System.Drawing.Size(95, 16);
            this.lblSegments.TabIndex = 0;
            this.lblSegments.Text = "Segments Done:";
            // 
            // lstThumbs
            // 
            this.lstThumbs.BackColor = System.Drawing.SystemColors.ControlDarkDark;
            this.lstThumbs.ForeColor = System.Drawing.SystemColors.Window;
            this.lstThumbs.Location = new System.Drawing.Point(12, 12);
            this.lstThumbs.Name = "lstThumbs";
            this.lstThumbs.Size = new System.Drawing.Size(551, 335);
            this.lstThumbs.TabIndex = 8;
            this.lstThumbs.UseCompatibleStateImageBehavior = false;
            this.lstThumbs.SelectedIndexChanged += new System.EventHandler(this.lstThumbs_SelectedIndexChanged);
            // 
            // btnStop
            // 
            this.btnStop.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnStop.Location = new System.Drawing.Point(406, 538);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(157, 27);
            this.btnStop.TabIndex = 9;
            this.btnStop.Text = "Stop Analysis";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // btnCarve
            // 
            this.btnCarve.Enabled = false;
            this.btnCarve.Font = new System.Drawing.Font("Century Gothic", 9F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCarve.Location = new System.Drawing.Point(12, 538);
            this.btnCarve.Name = "btnCarve";
            this.btnCarve.Size = new System.Drawing.Size(157, 27);
            this.btnCarve.TabIndex = 10;
            this.btnCarve.Text = "Reconstruct Files";
            this.btnCarve.UseVisualStyleBackColor = true;
            this.btnCarve.Click += new System.EventHandler(this.btnCarve_Click);
            // 
            // Analysis
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(96F, 96F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Dpi;
            this.ClientSize = new System.Drawing.Size(575, 577);
            this.Controls.Add(this.btnCarve);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.lstThumbs);
            this.Controls.Add(this.grpProcessed);
            this.Controls.Add(this.grpGPUActivity);
            this.Controls.Add(this.lblProcess);
            this.Controls.Add(this.lblProgress);
            this.Controls.Add(this.lblHeader);
            this.Controls.Add(this.pbProgress);
            this.Font = new System.Drawing.Font("Century Gothic", 8.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.Icon = ((System.Drawing.Icon)(resources.GetObject("$this.Icon")));
            this.Margin = new System.Windows.Forms.Padding(3, 4, 3, 4);
            this.MaximizeBox = false;
            this.Name = "Analysis";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "File Analysis";
            this.FormClosing += new System.Windows.Forms.FormClosingEventHandler(this.Carve_FormClosing);
            this.Load += new System.EventHandler(this.Carve_Load);
            this.grpGPUActivity.ResumeLayout(false);
            this.grpGPUActivity.PerformLayout();
            this.grpProcessed.ResumeLayout(false);
            this.grpProcessed.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.ProgressBar pbProgress;
        private System.Windows.Forms.Label lblHeader;
        private System.Windows.Forms.Label lblProgress;
        private System.Windows.Forms.Label lblProcess;
        private System.Windows.Forms.GroupBox grpGPUActivity;
        private System.Windows.Forms.GroupBox grpProcessed;
        private System.Windows.Forms.Label lblFound;
        private System.Windows.Forms.Label lblSegments;
        private System.Windows.Forms.TableLayoutPanel tblGPU;
        private System.Windows.Forms.Label lblTimeRemaining;
        private System.Windows.Forms.Label lblTimeElapsed;
        private System.Windows.Forms.Label lblFoundValue;
        private System.Windows.Forms.Label lblSegmentsValue;
        private System.Windows.Forms.Label lblTimeRemainingValue;
        private System.Windows.Forms.Label lblTimeElapsedValue;
        private System.Windows.Forms.ListView lstThumbs;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.Button btnCarve;
    }
}