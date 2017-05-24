$_bashExe = "c:\cygwin\bin\bash.exe"
$_DdsmRoot = "C:\Code\Data\DATA698-ResearchProj\ddsm"
#$_Output = "C:\Code\Python\DATA698-ResearchProj\data\ddsm\myOutput"
$_OutputRoot = "C:\Users\Dan\Dropbox\DATA698-S17\data\ddsm"
$_PngOutput = $_OutputRoot + "\png"
$_PgmOutput = $_OutputRoot + "\pgm"
$_jpegPath = "/cygdrive/c/code/Data/DATA698-ResearchProj/ddsm/jpeg.exe"
$_ddsmraw2pnmPath = "/cygdrive/c/code/Data/DATA698-ResearchProj/ddsm/ddsmraw2pnm.exe"
$_pnmtopngPath = "/cygdrive/c/cygwin/bin/pnmtopng.exe"
$_pamtopnmPath = "/cygdrive/c/cygwin/bin/pamtopnm.exe"

#  pamtopnm.exe /cygdrive/c/Code/Python/DATA698-ResearchProj/data/ddsm/normals/normal_01/case0002/A_0002_1.LEFT_CC.LJPEG.1-ddsmraw2pnm.pnm > output.pgm


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
    $obj | Add-Member -Type NoteProperty -Name OutputFile -Value "";
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

function Get-OverlayMetaData
{
    param (
        $imgFile
    )

    $data = @{}
    $overlayFilename = ""
    $nameParts = $imgFile.Name.Split(".")
    for($i = 0; $i -lt $nameParts.Length - 1; $i++)
    {
        $overlayFilename += $nameParts[$i] + "."
    }
    $overlayFilename += "OVERLAY"

    $overlayFullPath = Join-Path -Path $imgFile.Directory.FullName -ChildPath $overlayFilename
    if([System.IO.File]::Exists($overlayFullPath))
    {

        $overLayData = Get-Content $overlayFullPath
        $lines = $overLayData.Split([Environment]::NewLine) 
        $names = @("LESION_TYPE", "PATHOLOGY")
    
        foreach($line in $lines)
        {
            
            $lineItems = $line.Split(" ")
            if($names.Contains($lineItems[0]))
            { 
                $data[$lineItems[0]] = $lineItems[1]
                # take the first abnormality that is malignant
                if($lineItems[0] -eq "PATHOLOGY" -and
                   $lineItems[1] -eq "MALIGNANT")
                {
                    break
                }
            }

            # Log multiple abnormalities FYI
            if($lineItems[0] -eq "TOTAL_ABNORMALITIES")
            {
                if($lineItems[1] -gt 1)
                {
                    ($imgFile.Name + ": " + $line) | Out-Host 
                }
            }

        }
    }
    
    return $data
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
    $obj | Add-Member -Type NoteProperty -Name SubFolder -Value "";
    $obj | Add-Member -Type NoteProperty -Name Pathology -Value "";
    $obj | Add-Member -Type NoteProperty -Name LesionType -Value "";
    

    return $obj
}

$resultFileType = "pgm"
if($resultFileType -eq "pgm")
{
    $_Output = $_PgmOutput
}
elseif($resultFileType -eq "png")
{
    $_Output = $_PngOutput
}

# Get the list of LJPEG files to decode
$files = Get-ChildItem -Path $_DdsmRoot -Filter "*.LJPEG" -Recurse

# If you want to only work on a single file, fill in the path below and uncomment the line.
#$files = @(Get-Item -Path "C:\Code\Data\DATA698-ResearchProj\ddsm\cancers\cancer_07\case1160\A_1160_1.RIGHT_MLO.LJPEG")


# Load CSV meta data if available.
$lastSave = (Get-Date).AddDays(-1)
$CsvOutput = Join-Path -Path $_Output -ChildPath "Ddsm.csv"
$csvData = New-Object System.Collections.ArrayList
if ([System.IO.File]::Exists($CsvOutput))
{
    # Load the existing CSV
    $importedCsvData = Import-Csv $CsvOutput
    $csvData.AddRange($importedCsvData)

    # Check for new/added properties that weren't in the first pass csv creation.
    $bAddAbType = $false
    $bAddScanner = $false
    $bAddPathology = $false
    $bAddLesionType = $false
    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "AbType"}))
    {
        $bAddAbType = $true
    }

    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "Scanner"}))
    {
        $bAddScanner = $true
    }

    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "Pathology"}))
    {
        $bAddPathology = $true
    }

    if(-not [bool]($csvData[0].psobject.Properties | where { $_.Name -eq "LesionType"}))
    {
        $bAddLesionType = $true
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

        if($bAddPathology)
        {
            $csvData[$i]  | Add-Member -Type NoteProperty -Name Pathology -Value "";
        }

        if($bAddLesionType)
        {
            $csvData[$i]  | Add-Member -Type NoteProperty -Name LesionType -Value "";
        }
    }
}



$icsList = @{}
$fc = 0
# Loop through them
foreach($f in $files)
{
    $fc += 1
    Write-Progress -id 202 -Activity $f.Name -Status ($files.Length - $fc) -PercentComplete (100 * ($fc / $files.Length))

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
    $winLjpeg1 = ($f.FullName + ".1")
    $cygFile = Build-CygFile $f.FullName 
    $ljpeg1 = ($cygFile + ".1")
    $winPnm = ($winLjpeg1 + "-ddsmraw2pnm.pnm")
    $pnm = ($ljpeg1 + "-ddsmraw2pnm.pnm")

    # Look up the img data for this image
    $fileParts = $f.Name.Split(".")
    $firstParts = $fileParts.Split("_")
    $caseNum = $firstParts[1]
    $subFolder = [math]::floor($caseNum / 1000)
    $imgData = $icsData.ImageData[$fileParts[1]]
    if ($resultFileType -eq "png")
    {
        $png = $f.Name + ".png"
        $imgData.OutputFile = $png
        $winFile = (Join-Path -Path $_PngOutput -ChildPath (Join-Path $subFolder -ChildPath $png))
    }
    elseif ($resultFileType -eq "pgm")
    {
        $pgm = $f.Name + ".pgm"
        $imgData.OutputFile = $pgm
        $winFile = (Join-Path -Path $_PgmOutput -ChildPath (Join-Path $subFolder -ChildPath $pgm))
    }

    # Ensure subfolder exists
    $_FileOutput = Join-Path -Path $_Output -ChildPath $subFolder
    if(-not (Test-Path -Path $_FileOutput -PathType Container))
    {
        New-Item -ItemType Directory -Force -Path $_FileOutput
    }
    

    # Does the PGM/PNG already exist? If not, then go through conversion process.
    $theFile = Get-Item $winFile
    if ((![System.IO.File]::Exists($winFile) -or $theFile.Length -eq 0))
    {
        if (![System.IO.File]::Exists($winLjpeg1))
        {
            # JPEG command
            ("Converting " + $f.Name + " to LJPEG.1...") | Out-Host
            $cmdJpeg = ("$_jpegPath -d -s ") + $cygFile
            $result = Execute-BashCommand -Command $cmdJpeg
        }


        if (![System.IO.File]::Exists($winPnm))
        {
            # DDSMRAW2PNM command
            ("Converting $ljpeg1 to PNM...") | Out-Host 
            $cmdDdsm = ("$_ddsmraw2pnmPath ") + $ljpeg1 + " " + $imgData.Rows + " " + $imgData.Columns + " " + $icsData.Digitizer.ToLower()
            $result = Execute-BashCommand -Command $cmdDdsm
        }

        # Only create the PNG/PGM if the PNM from prior step exists.
        if([System.IO.File]::Exists($winPnm))
        {
            # PNMTOXyz
            ("Converting $pnm to $resultFileType...") | Out-Host 
            $outputFile = Build-CygFile $winFile
            $cmdXyz = ("$_pamtopnmPath ") + $pnm + " > " + $outputFile
            $result = Execute-BashCommand -Command $cmdXyz
        }

        # CSV Data
        #$csvLine = New-DdsmCsvLine
        #$csvLine.Name = $png
        #$csvLine.Type = $icsData.Name.Substring(0,1)
        #$csvLine.AbType = $parentDir
        #$csvLine.Scanner = $icsData.Digitizer.ToLower()

        #$csvData.Add($csvLine) > $null
    }

    # Only update the CSV if we have an output PNG/PGM
    if([System.IO.File]::Exists($winFile))
    {
        $line = $csvData | where {$_.Name -eq $imgData.OutputFile}
        if($line -eq $null)
        {
            $line = New-DdsmCsvLine
            $line.Name = $imgData.OutputFile
            $csvData.Add($line) > $null
        }
        
        $line.Type = $icsData.Name.Substring(0,1)
        $line.AbType = $parentDir
        $line.Scanner = $icsData.Digitizer.ToLower()
        $line.SubFolder = $subFolder
        
        $ovData = Get-OverlayMetaData $f
        if($ovData.Count -gt 0)
        {
            $line.Pathology = $ovData["PATHOLOGY"]
            $line.LesionType = $ovData["LESION_TYPE"]
        }

        # Only save every so often. Helps with Dropbox syncing.
        $now = Get-Date 
        $timeDiff = New-TimeSpan $lastSave $now
        if($timeDiff.TotalSeconds -gt 60)
        {
            # Resave the CSV
            $csvData | Export-Csv -Path $CsvOutput
            $lastSave = Get-Date 
        }
    } 

    
}

