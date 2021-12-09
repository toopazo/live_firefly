
echo "Download logs from pi@10.42.0.103 --------------------------"

# scp [OPTION] [user@]SRC_HOST:]file1 [user@]DEST_HOST:]file2
scp pi@10.42.0.103:/home/pi/live_firefly/logs/* logs

# tzo4@E5-AERO-RHEA has /home/tzo4/.ssh/id_rsa.pub equal to
# ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCrMkehrJ8f+GRSDfBznQpwzrUgqV9ljpVfIzrTJ/gQNAqa3FO19vsUr3IDE1uAMvMKIdY0FhH3CpiNkQKEzLgRcWmtk164LcxcreuxS9JEB/7NW4wBMs5GtRXqXVd4gc3OOE6+qMfEs+3JgsNtcVIPATVvswUcR6dpeJoLmK/T5uv7Fr8yYgGE8eNij5v0tOsA6xBNXnFSNDcp6x6f5xhLFpEUwBpcr+RLAwMMkSFbZ740G/305s0rKx46W0f0UFe1i4Xz9doS6ryW8eiN/N9GCUdlXl67SN48JveuZBwVVaBiGXDjnaui8m4CE5T8QNkWQMxj/fK7zzumvZoBcnUPJhjklA2zxy2By2qSX8WJNq9swrDfiugYOWUGb0ZhKx3PMTEwiKviDkWdbodjFGNSPHwr/RvXSM3CrdhJ/5xY3x6l41l69ejpWdPP+zTZ18SBBjRc8ewhEC3WuJlLX7voy4DxCCTlenfYA8Ls2trfrZECXdVzvf2NGvOzQlYoOlmsOw/DGf9vR4UjO6hR49Ls9WiQnRsBPfyng29MC2avWw6Z34Ti6WkvJdYrF6z8nCGE6AborH0CpH9XtTaTGxfaj+Fy5xykNP28EPKmhWd+4Cc6H7QlnT6E440OpVWUe2k81iQ6NGYh2rPFjuHMdoGMPXnonrntIf8YrLUtEVBViw== toopazo@protonmail.com
# This script assumes it is already uploaded to pi@raspberry:/home/pi/.ssh/authorized_keys
