clear
echo "---------------SUM OF N NUMBER  IN SHELL SCRIPT------------"
echo -n "enter  nth numbers's value:"
read digit
t=1
total=0
while test $t -le $digit
do
total=`expr $total + $t`
t=`expr $t + 1`
done
echo "sum of $digit: $total "


echo "---------------FACTORIAL OF N NUMBER  IN SHELL SCRIPT------------"
read -p "Enter a positive integer: " num

fact=1
for (( i=1; i<=num; i++ ))
do
  fact=$((fact * i))
done

echo "Factorial of $num is: $fact"

