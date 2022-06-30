using System;
using System.Management;
using System.Windows.Forms;

namespace OpenForensics
{
    public partial class PhysicalDriveDialog : Form
    {
        public PhysicalDriveDialog()
        {
            InitializeComponent();
        }

        public string physicalDrive = "";

        private void PhysicalDriveDialog_Load(object sender, EventArgs e)
        {
            ManagementObjectSearcher mosDisks = new ManagementObjectSearcher("SELECT * FROM Win32_DiskDrive WHERE MediaType IS NOT NULL");

            foreach (ManagementObject moDisk in mosDisks.Get())
                cmbHdd.Items.Add(moDisk["DeviceID"].ToString());

            cmbHdd.SelectedIndex = 0;
        }

        private void cmbHdd_SelectedIndexChanged(object sender, EventArgs e)
        {
            ManagementObjectSearcher mosDisks = new ManagementObjectSearcher("SELECT * FROM Win32_DiskDrive WHERE DeviceID = '" + cmbHdd.SelectedItem.ToString().Replace("\\", "\\\\") + "'");
            foreach (ManagementObject moDisk in mosDisks.Get())
            {
                try
                {
                    lblType.Text = "Type: " + moDisk["MediaType"].ToString();
                    lblModel.Text = "Model: " + moDisk["Model"].ToString();
                    lblSerial.Text = "Serial: " + moDisk["SerialNumber"].ToString();
                    lblInterface.Text = "Interface: " + moDisk["InterfaceType"].ToString();
                    lblCapacity.Text = "Capacity: " + Math.Round(((((double)Convert.ToDouble(moDisk["Size"]) / 1024) / 1024) / 1024), 2) + " GB";
                    lblPartitions.Text = "Partitions: " + moDisk["Partitions"].ToString();
                    lblFirmware.Text = "Firmware: " + moDisk["FirmwareRevision"].ToString();
                    lblCylinders.Text = "Cylinders: " + moDisk["TotalCylinders"].ToString();
                    lblSectors.Text = "Sectors: " + moDisk["TotalSectors"].ToString();
                    lblHeads.Text = "Heads: " + moDisk["TotalHeads"].ToString();
                    lblTracks.Text = "Tracks: " + moDisk["TotalTracks"].ToString();
                    lblBytesPerSect.Text = "Bytes per Sector: " + moDisk["BytesPerSector"].ToString();
                    lblSectorsPerTrack.Text = "Sectors per Track: " + moDisk["SectorsPerTrack"].ToString();
                    lblTracksPerCyl.Text = "Tracks per Cylinder: " + moDisk["TracksPerCylinder"].ToString();
                    physicalDrive = moDisk["DeviceID"].ToString();
                }
                catch //(Exception ex)
                {
                    //MessageBox.Show(ex.ToString());
                    lblType.Text = "Type: ";
                    lblModel.Text = "Model: ";
                    lblSerial.Text = "Serial: ";
                    lblInterface.Text = "Interface: ";
                    lblCapacity.Text = "Capacity: ";
                    lblPartitions.Text = "Partitions: ";
                    lblFirmware.Text = "Firmware: ";
                    lblCylinders.Text = "Cylinders: ";
                    lblSectors.Text = "Sectors: ";
                    lblHeads.Text = "Heads: ";
                    lblTracks.Text = "Tracks: ";
                    lblBytesPerSect.Text = "Bytes per Sector: ";
                    lblSectorsPerTrack.Text = "Sectors per Track: ";
                    lblTracksPerCyl.Text = "Tracks per Cylinder: ";
                    lblType.Text = "Type: Drive Not Connected!";
                    physicalDrive = "";
                }
            }
        }

        private void btnSelect_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.OK;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            this.DialogResult = DialogResult.Cancel;
        }
    }
}
