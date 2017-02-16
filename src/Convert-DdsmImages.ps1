$_bashExe = "c:\cygwin\bin\bash.exe"
$_DdsmRoot = "C:\Code\Python\DATA698-ResearchProj\data\ddsm"
$_Output = "C:\Code\Python\DATA698-ResearchProj\data\ddsm\myOutput"
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

    return $obj
}



# Get the list of LJPEG files to decode
$files = Get-ChildItem -Path $_DdsmRoot -Filter "*.LJPEG" -Recurse

#Out-Host $files.Count
$csvData = New-Object System.Collections.ArrayList
$icsList = @{}
# Loop through them
foreach($f in $files)
{
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

    

    # JPEG command
    ("Converting " + $f.Name + " to LJPEG.1...") | Out-Host
    $cygFile = Build-CygFile $f.FullName 
    
    $cmdJpeg = ("$_jpegPath -d -s ") + $cygFile
    $result = Execute-BashCommand -Command $cmdJpeg

    # Look up the img data for this image
    $fileParts = $f.Name.Split(".")
    $imgData = $icsData.ImageData[$fileParts[1]]

    # DDSMRAW2PNM command
    $ljpeg1 = ($cygFile + ".1")
    ("Converting $ljpeg1 to PNM...") | Out-Host 
    $cmdDdsm = ("$_ddsmraw2pnmPath ") + $ljpeg1 + " " + $imgData.Rows + " " + $imgData.Columns + " " + $icsData.Digitizer.ToLower()
    $result = Execute-BashCommand -Command $cmdDdsm

    # PNMTOPNG
    $pnm = ($ljpeg1 + "-ddsmraw2pnm.pnm")
    $png = $f.Name + ".png"
    $outputPng = Build-CygFile  (Join-Path -Path $_Output -ChildPath $png)

    ("Converting $pnm to PNG...") | Out-Host 
    $cmdPng = ("$_pnmtopngPath -verbose ") + $pnm + " > " +$outputPng
    $result = Execute-BashCommand -Command $cmdPng
    $imgData.Png = $png

    # CSV Data
    $csvLine = New-DdsmCsvLine
    $csvLine.Name = $png
    $csvLine.Type = $icsData.Name.Substring(0,1)

    $csvData.Add($csvLine) > $null
    $CsvOutput = Join-Path -Path $_Output -ChildPath "Ddsm.csv"
    $csvData | Export-Csv -Path $CsvOutput
}

