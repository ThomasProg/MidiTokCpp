# MidiTokCpp
A C++ version of MIDITok, for fast tokenizer use in C++

To check if functions have been exported correctly:
```
dumpbin.exe -headers MidiTokCpp.lib | findstr /c:"  Symbol name  :" > foo-exports.txt
```