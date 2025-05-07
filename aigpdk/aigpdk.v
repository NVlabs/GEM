// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
module AND2_00_0 (A, B, Y);
input  A ;
input  B ;
output Y ;

   assign Y = A & B;

endmodule // AND2_00_0

module AND2_01_0 (A, B, Y);
input  A ;
input  B ;
output Y ;

   assign Y = A & ~B;

endmodule // AND2_01_0

module AND2_10_0 (A, B, Y);
input  A ;
input  B ;
output Y ;

   assign Y = ~A & B;

endmodule // AND2_10_0

module AND2_11_0 (A, B, Y);
input  A ;
input  B ;
output Y ;

   assign Y = ~A & ~B;

endmodule // AND2_11_0

module AND2_11_1 (A, B, Y);
input  A ;
input  B ;
output Y ;

   assign Y = A | B;

endmodule // AND2_11_1

module INV (A, Y);
input  A ;
output Y ;

   not (Y, A);

endmodule // INV

module BUF (A, Y);
input  A ;
output Y ;

   assign Y = A;

endmodule // BUF

primitive udp_dff (out, in, clk, clr, set, NOTIFIER);
   output out;
   input  in, clk, clr, set, NOTIFIER;
   reg    out;

   table

// in  clk  clr   set  NOT  : Qt : Qt+1
//
   0  r   ?   0   ?   : ?  :  0  ; // clock in 0
   1  r   0   ?   ?   : ?  :  1  ; // clock in 1
   1  *   0   ?   ?   : 1  :  1  ; // reduce pessimism
   0  *   ?   0   ?   : 0  :  0  ; // reduce pessimism
   ?  f   ?   ?   ?   : ?  :  -  ; // no changes on negedge clk
   *  b   ?   ?   ?   : ?  :  -  ; // no changes when in switches
   ?  ?   ?   1   ?   : ?  :  1  ; // set output
   ?  b   0   *   ?   : 1  :  1  ; // cover all transistions on set
   1  x   0   *   ?   : 1  :  1  ; // cover all transistions on set
   ?  ?   1   0   ?   : ?  :  0  ; // reset output
   ?  b   *   0   ?   : 0  :  0  ; // cover all transistions on clr
   0  x   *   0   ?   : 0  :  0  ; // cover all transistions on clr
   ?  ?   ?   ?   *   : ?  :  x  ; // any notifier changed

   endtable
endprimitive // udp_dff

primitive udp_tlat (out, in, enable, clr, set, NOTIFIER);

   output out;
   input  in, enable, clr, set, NOTIFIER;
   reg    out;

   table

// in  enable  clr   set  NOT  : Qt : Qt+1
//
   1  1   0   ?   ?   : ?  :  1  ; //
   0  1   ?   0   ?   : ?  :  0  ; //
   1  *   0   ?   ?   : 1  :  1  ; // reduce pessimism
   0  *   ?   0   ?   : 0  :  0  ; // reduce pessimism
   *  0   ?   ?   ?   : ?  :  -  ; // no changes when in switches
   ?  ?   ?   1   ?   : ?  :  1  ; // set output
   ?  0   0   *   ?   : 1  :  1  ; // cover all transistions on set
   1  ?   0   *   ?   : 1  :  1  ; // cover all transistions on set
   ?  ?   1   0   ?   : ?  :  0  ; // reset output
   ?  0   *   0   ?   : 0  :  0  ; // cover all transistions on clr
   0  ?   *   0   ?   : 0  :  0  ; // cover all transistions on clr
   ?  ?   ?   ?   *   : ?  :  x  ; // any notifier changed

   endtable
endprimitive // udp_tlat

module DFF (CLK, D, Q);
input  CLK ;
input  D ;
output Q ;
reg NOTIFIER ;

   udp_dff (DS0000, D, CLK, 1'B0, 1'B0, NOTIFIER);
   not (P0002, DS0000);
   buf (Q, DS0000);

endmodule // DFF

module DFFSR (CLK, D, R, S, Q);
input  CLK ;
input  D ;
input  R ;
input  S ;
output Q ;
reg NOTIFIER ;

   not (I0_CLEAR, R);
   not (I0_SET, S);
   udp_dff (P0003, D_, CLK, I0_SET, I0_CLEAR, NOTIFIER);
   not (D_, D);
   not (P0002, P0003);
   buf (Q, P0002);
   and (\D&S , D, S);
   not (I7_out, D);
   and (\~D&R , I7_out, R);
   and (\S&R , S, R);

endmodule // DFFSR

// module LATCH (CLK, D, Q);
// input  CLK ;
// input  D ;
// output Q ;
// reg NOTIFIER ;

//    udp_tlat (DS0000, D, CLK, 1'B0, 1'B0, NOTIFIER);
//    not (P0000, DS0000);
//    buf (Q, DS0000);

// endmodule

module CKLNQD (CP, E, Q);
   (* gated_clock = "true" *) input CP;
   input E;
   output Q;
   reg    QD;
   always @* begin
      if (~CP) QD <= E;
   end
   assign Q = CP & QD;
endmodule
