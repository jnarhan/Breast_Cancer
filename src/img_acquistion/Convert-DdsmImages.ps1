$_bashExe = "c:\cygwin\bin\bash.exe"
$_DdsmRoot = "C:\Code\Python\DATA698-ResearchProj\data\ddsm"
#$_Output = "C:\Code\Python\DATA698-ResearchProj\data\ddsm\myOutput"
$_Output = "C:\Users\Dan\Dropbox\DATA698-ResearchProj\data\ddsm\png"
$_jpegPath = "/cygdrive/c/code/python/DATA698-ResearchProj/data/ddsm/jpeg.exe"
$_ddsmraw2pnmPath = "/cygdrive/c/code/python/DATA698-ResearchProj/data/ddsm/ddsmraw2pnm.exe"
$_pnmtopngPath = "/cygdrive/c/cygwin/bin/pnmtopng.exe"

function Execute-BashCommand
{
    param (
        [string] $Command
    )
    $StdOutLog = "BashCommand-stdout.txt"
    $bashArgs =  @("-c", ('"' + $Command + '"'))

    $proc = Start-Process $_bashExe -ArgumentList $bashArgs -NoNewWindow -PassThru -RedirectStandardOutput $StdOutLog
    if( $null -ne $proc)
    {
        $proc.WaitForExit()
        $result = Get-Content  $StdOutLog
    }

    return $result
}

function New-IcsData
{
    $obj = New-Object PSObject
    $obj | Add-Member -Type NoteProperty -Name Name -Value "";
    $obj | Add-Member -Type NoteProperty -Name Digitizer -Value "";
    $obj | Add-Member -Type NoteProperty -Name ImageData -Value @{};
    return $obj
}

function New-IcsImageData
{
    $obj = New-Object PSObject
    $obj | Add-Member -Type NoteProperty -Name Name -Value "";
    $obj | Add-Member -Type NoteProperty -Name Rows -Value 0;
    $obj | Add-Member -Type NoteProperty -Name Columns -Value 0;
    $obj | Add-Member -Type NoteProperty -Name Png -Value "";
    return $obj
}

function Get-DdsmDigitizer
{
    param (
        $ics
    )

    $digitizer = "Unknown"
    $icsLines = $ics.Split([Environment]::NewLine) 
    #$icsLines[9] | Out-Host

    $line = $icsLines[9]
    if(-not $line.StartsWith("DIGITIZER"))
    {
        Out-Host "Digitizer not found!"
    }
    else
    {
        $items = $line.Split(" ")
        $digitizer = $items[1]
    }
    
    return $digitizer
}

function Get-IcsImageMetaData
{
    param (
        $ics,
        $icsData
    )

    $icsLines = $ics.Split([Environment]::NewLine) 
    $imgNames = @("LEFT_CC", "LEFT_MLO", "RIGHT_CC", "RIGHT_MLO")

    foreach($line in $icsLines)
    {
        $lineItems = $line.Split(" ")
        if($imgNames.Contains($lineItems[0]))
        { 
            $icsData.ImageData[$lineItems[0]] = New-IcsImageData
            $icsData.ImageData[$lineItems[0]].Name = $lineItems[0]
            $icsData.ImageData[$lineItems[0]].Rows = $lineItems[2]
            $icsData.ImageData[$lineItems[0]].Columns = $lineItems[4]
        }
        
    }
    
    return $icsData
}

function Build-CygFile
{
    param (
        $file
    )

    $cygFile = "/cygdrive/" + $file.Replace("\", "/").Replace(":", "")


    return $cygFile
}

function New-DdsmCsvLine
{
    $obj = New-Object PSObject
    $obj | Add-Member -Type NoteProperty -Name Name -Value "";
    $obj | Add-Member -Type NoteProperty -Name Type -Value "";
    $obj | Add-Member -Type NoteProperty -Name AbType -Value "";
    $obj | Add-Member -Type NoteProperty -Name Scanner -Value "";

    return $obj
}



# Get the list of LJPEG files to decode
$files = Get-ChildItem -Path $_DdsmRoot -Filter "*.LJPEG" -Recurse

# Load CSV meta data if available.
$CsvOutput = Join-Path -Path $_Output -ChildPath "Ddsm.csv"
$csvData = New-Object System.Collections.ArrayList
if ([System.IO.File]::Exists($CsvOutput))
{
    # Load the existing CSV
    $importedCsvData = Import-Csv $CsvOutput
    $csvData.AddRange($importedCsvData)
    $lastSave = (Get-Date).AddDays(-1)

    # Check for new/added properties that weren't in the first pass csv creation.
    $bAddAbType = $false
    $bAddScanner = $false
    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "AbType"}))
    {
        $bAddAbType = $true
    }

    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "Scanner"}))
    {
        $bAddScanner = $true
    }

    # Add the missing properties
    for($i = 0; $i -lt $csvData.Count; $i++)
    {
        if($bAddAbType)
        {
            $csvData[$i]  | Add-Member -Type NoteProperty -Name AbType -Value "";
        }

        if($bAddScanner)
        {
            $csvData[$i]  | Add-Member -Type NoteProperty -Name Scanner -Value "";
        }
    }
}

$icsList = @{}
$fc = 0
# Loop through them
foreach($f in $files)
{
    $fc += 1.0
    Write-Progress -Activity $f.Name -PercentComplete ($fc / $files.Length)

    $parentDir = $f.Directory.Parent.Parent.Name
    $icsFile = Get-ChildItem -Path $f.DirectoryName -Filter "*.ics"
    # Have we already loaded this ICS file?
    if(-not $icsList.ContainsKey($icsFile.Name))
    {
        # No - load it
        $ics = Get-Content $icsFile.FullName
        $icsData = New-IcsData
        $icsData = Get-IcsImageMetaData $ics $icsData
        $icsData.Name = $icsFile.Name
        $icsData.Digitizer = Get-DdsmDigitizer $ics
        $icsList.Add($icsFile.Name, $icsData)
    }

    # Fetch the ICS data structure
    $icsData = $icsList[$icsFile.Name]

    # Create image conversion file names
    $cygFile = Build-CygFile $f.FullName 
    $ljpeg1 = ($cygFile + ".1")
    $pnm = ($ljpeg1 + "-ddsmraw2pnm.pnm")
    $png = $f.Name + ".png"
    $winPng = (Join-Path -Path $_Output -ChildPath $png)
    $outputPng = Build-CygFile $winPng

    # Does the PNG already exist? If not, then go through conversion process.
    $filePng = Get-Item $winPng
    if (![System.IO.File]::Exists($winPng) -or $filePng.Length -eq 0)
    {
        # JPEG command
        ("Converting " + $f.Name + " to LJPEG.1...") | Out-Host
        $cmdJpeg = ("$_jpegPath -d -s ") + $cygFile
        $result = Execute-BashCommand -Command $cmdJpeg

        # Look up the img data for this image
        $fileParts = $f.Name.Split(".")
        $imgData = $icsData.ImageData[$fileParts[1]]

        # DDSMRAW2PNM command
        ("Converting $ljpeg1 to PNM...") | Out-Host 
        $cmdDdsm = ("$_ddsmraw2pnmPath ") + $ljpeg1 + " " + $imgData.Rows + " " + $imgData.Columns + " " + $icsData.Digitizer.ToLower()
        $result = Execute-BashCommand -Command $cmdDdsm

        # PNMTOPNG
        ("Converting $pnm to PNG...") | Out-Host 
        $cmdPng = ("$_pnmtopngPath -verbose ") + $pnm + " > " +$outputPng
        $result = Execute-BashCommand -Command $cmdPng
        $imgData.Png = $png

        # CSV Data
        #$csvLine = New-DdsmCsvLine
        #$csvLine.Name = $png
        #$csvLine.Type = $icsData.Name.Substring(0,1)
        #$csvLine.AbType = $parentDir
        #$csvLine.Scanner = $icsData.Digitizer.ToLower()

        #$csvData.Add($csvLine) > $null
    }

    $line = $csvData | where {$_.Name -eq $png}
    if($line -eq $null)
    {
        $line = New-DdsmCsvLine
        $line.Name = $png
        $line.Type = $icsData.Name.Substring(0,1)
        $csvData.Add($line) > $null
    }
        
    $line.AbType = $parentDir
    $line.Scanner = $icsData.Digitizer.ToLower()

    # Only save every so often. Helps with Dropbox syncing.
    $now = Get-Date 
    $timeDiff = New-TimeSpan $lastSave $now
    if($timeDiff.TotalSeconds -gt 30)
    {
        # Resave the CSV
        $csvData | Export-Csv -Path $CsvOutput
        $lastSave = Get-Date 
    }    

    
}

