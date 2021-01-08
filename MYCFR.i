%module MYCFR
%{
    #include "game.h"
    #include "oracle.h"
    #include "cfr.h"
    #include "calculator.h"
%}
namespace std {
%template(Line)  vector < int >;
    %template(Array) vector < vector < int> >;
}   
%include "std_vector.i"
%include "game.h"
%include "oracle.h"
%include "cfr.h"
%include "calculator.h"