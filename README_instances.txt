#cust #prod #vehi Q #predictError


X(0) Y(0)


for each prod
   I0(supplier, prod)
endfor

Capacity(supplier)

for each prod
   h(0, prod)
endfor

for each vehi
   b(vehi)
endfor


for each cust
   index X(i) Y(i)

   for each prod
      I0(cust, prod)
   endfor

   for each prod
      d_actual(cuts, prod)
   endfor

   for each prod
      d_predict(cuts, prod)
   endfor

   Capacity(customer)

   for each prod
     h(cust, prod)
   endfor

   for each prod
     p(cust, prod)
   endfor

   for each prod
     v(cust, prod)
   endfor

endfor