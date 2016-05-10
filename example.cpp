#include <stdio.h>
#include "uni10.hpp"

using namespace std;
using namespace uni10;

int main(int argc, char **argv){

  // rsp_label == origin labels
  int rsp_label[] = {0, 1, 2, 3, 4, 5, 6, 7};
  vector<Bond> bonds(8, Bond(BD_IN, 3));
  UniTensor A(bonds);
  A.randomize();
  //cout << A << endl;   // print elements
  A.permute(rsp_label, 2);
  A.printGraphy();
  //cout << A << endl;
  A.permute(4);
  A.printGraphy();

  // rsp_label != ori_label.
  
  int rsp_labelD[] = {7,3,4,2,1,0,6,5};
  A.permute(rsp_labelD, 2);
  A.printGraphy();
  //cout << A << endl;
  A.permute(rsp_labelD, 4);
  A.printGraphy();
  //cout << A << endl;
  A.permute(8);
  A.printGraphy();
  return 0;

}
