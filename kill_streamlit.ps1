Get-WmiObject Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match "streamlit" } | ForEach-Object {
    Write-Host "Killing process $($_.ProcessId)"
    Stop-Process -Id $_.ProcessId -Force
}
